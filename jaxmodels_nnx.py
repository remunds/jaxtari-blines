
from functools import reduce
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from flax.nnx import initializers
from omegaconf import OmegaConf as oc
from utils.jaxutils import symlog, symexp


prod = lambda lst: reduce(lambda x, y: x*y, lst, 1)

ACTIVATIONS = {
    'relu' : nnx.relu,
    'silu': nnx.silu,
    'elu': nnx.elu,
    'tanh': nnx.tanh,
    'sigmoid': nnx.sigmoid,
    'softmax': nnx.softmax,
    'symlog': symlog,
    'symexp': symexp,
    'none': lambda x: x
}

NORMS = {
    'rms': nnx.RMSNorm,
    'layer': nnx.LayerNorm
}

class GRUCell(nnx.Module):
    def __init__(
        self,
        rngs : nnx.Rngs,
        input_dim : int,
        hidden_dim : int,
        bias : bool = True,
        **kwargs
    ):
        self.weights_x = nnx.Linear(input_dim, hidden_dim*3, use_bias=bias, rngs=rngs)
        self.weights_h = nnx.Linear(hidden_dim, hidden_dim*3, use_bias=bias, rngs=rngs)

    def __call__(self, x, h):
        x_r, x_z, x_n = jnp.split(self.weights_x(x), 3, axis=-1)
        h_r, h_z, h_n = jnp.split(self.weights_h(h), 3, axis=-1)
        r = nnx.sigmoid(x_r + h_r)
        z = nnx.sigmoid(x_z + h_z)
        n = nnx.tanh(x_n + r * h_n)
        h = (1-z) * n + z * h
        return h, h

        
class BlockLinear(nnx.Module):
    def __init__(
            self,
            rngs : nnx.Rngs,
            groups : int,
            input_dim : int,
            output_dim : int,
            norm : bool = False,
            act : str = 'none'
        ):
        assert output_dim % groups == 0
        self.groups = groups
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = norm
        self.act = act
        _stack_linear = nnx.vmap(
            lambda key: nnx.Linear(input_dim, output_dim//groups, rngs=nnx.Rngs(key)),
            in_axes=0
        )
        self.mlps =_stack_linear(jax.random.split(rngs(), groups))
        # self.norm = NORMS['rms'](self.output_dim, rngs=rngs)
        self.norm_layer = NORMS['rms'](self.output_dim, rngs=rngs)
    
    def __call__(self, x):
        # batch_dims = x.shape[:-2]
        batch_dims = x.shape[:-2]
        x = x.reshape(-1, self.groups, self.input_dim)
        # new: pure jax grouped linear
        w = self.mlps.kernel.value
        x = jnp.einsum('bgi,gio->bgo', x, w)
        if getattr(self.mlps, 'bias', None) is not None:
            x = x + self.mlps.bias.value
        # x = nnx.vmap(lambda l, x: l(x), in_axes=(0, 1))(self.mlps, x) #(B, )
        # x = self.norm(x) if self.norm else x
        # x = x.swapaxes(0,1)
        x = x.reshape(*batch_dims, -1)
        x = self.norm_layer(x) if self.norm else x
        if self.act and self.act.lower() != 'none':
            x = getattr(nnx, self.act)(x)
        # return ACTIVATIONS[self.act](x)
        return x
    

class BlockGRUCell(nnx.Module):
    def __init__(
            self,
            rngs : nnx.Rngs,
            groups : int,
            input_dim : int,
            hidden_dim : int,
            **kwargs
        ):        
        self.groups = groups
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.linear_in = BlockLinear(
            rngs,
            groups,
            input_dim + hidden_dim // self.groups,
            hidden_dim,
            norm=True,
            act='silu'
        )
        self.linear_gru = BlockLinear(
            rngs,
            groups,
            hidden_dim // self.groups,
            hidden_dim*3,
            # norm=True
        )

    def __call__(self, x, h):
        batch_dims = x.shape[:-1]
        x = jnp.concatenate([h.reshape(*batch_dims, self.groups, -1), x[..., None, :].repeat(self.groups, -2)], axis=-1)
        x = self.linear_in(x).reshape(*batch_dims, self.groups, -1)
        x = self.linear_gru(x).reshape(*batch_dims, self.groups, -1)
        r, z, n = [g.reshape(*batch_dims, -1) for g in jnp.split(x, 3, axis=-1)]
        r = nnx.sigmoid(r)
        z = nnx.sigmoid(z)
        n = nnx.tanh(r * n)
        h = (1-z) * n + z * h
        return h, h


class NoisyLinear(nnx.Module):
    def __init__(self, input_dim, output_dim, sigma_zero=0.5, bias=True, rngs=nnx.Rngs(0), precision=jax.lax.Precision.DEFAULT):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_zero = sigma_zero
        self.bias = bias
        self.rngs = nnx.Rngs(rngs())

        sigma_init = self.sigma_zero / jnp.sqrt(input_dim)
        # self.kernel_mean = nnx.Param((jax.random.uniform(rngs(), (input_dim, output_dim))*2-1) / jnp.sqrt(input_dim))
        self.kernel_mean = nnx.Param(nnx.initializers.lecun_uniform()(rngs(), (input_dim, output_dim)))
        self.kernel_sigma = nnx.Param(nnx.initializers.constant(sigma_init)(rngs(), (input_dim, output_dim)))
        self.bias_mean = nnx.Param(nnx.initializers.constant(0.)(rngs(), (output_dim,))) if self.bias else 0.
        self.bias_sigma = nnx.Param(nnx.initializers.constant(sigma_init)(rngs(), (output_dim,))) if self.bias else 0.
        self.precision=precision

    def _scale_noise(self, noise):
        return jnp.sign(noise) * jnp.sqrt(jnp.abs(noise))
    
    def __call__(self, x, rng=None, mode='train'):
        if mode=='train':
            rng_1, rng_2 = jax.random.split(rng)#if rng else jax.random.split(self.rngs())
            eps_rows = self._scale_noise(jax.random.normal(rng_1,(self.input_dim,)))
            eps_cols = self._scale_noise(jax.random.normal(rng_2,(self.output_dim,)))
       
            kernel_noise = jnp.outer(eps_rows, eps_cols)
            bias_noise = eps_cols
            kernel = self.kernel_mean.value  + self.kernel_sigma.value * kernel_noise
            bias = self.bias_mean.value + self.bias_sigma.value * bias_noise
        else:
            kernel = self.kernel_mean.value
            bias = self.bias_mean.value
        bias = jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))

        return jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())), precision=self.precision) + bias


class NoisyMLP(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            input_dim: int = 1,
            hidden_dims: tuple =(64,),
            activation: str = 'silu',
            output_dim: int = 1,
            normalize: str = 'none',
            outact: str = 'none',
            sigma_zero : float = 0.5,
            dropout : float = 0.,
            **kwargs
        ):
        mlp_layers = (input_dim, *hidden_dims)
        layers = []
        normalize = None if normalize=='none' else normalize
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:]):
            layers.append(NoisyLinear(in_dim, out_dim, sigma_zero=sigma_zero, rngs=rngs))
            if dropout > 0:
                layers.append(nnx.Dropout(dropout, rngs=rngs))
            if normalize:
                layers.append(NORMS[normalize](out_dim, rngs=rngs))
            layers.append(ACTIVATIONS[activation])
            
        layers.append(NoisyLinear(mlp_layers[-1], output_dim, sigma_zero=sigma_zero, rngs=rngs))
        layers.append(ACTIVATIONS[outact])
        # self.layers = layers
        self.layers = nnx.List(layers)

    def __call__(self, x : jnp.ndarray, rng=None, mode='train'):
        for l in self.layers:
            if isinstance(l, NoisyLinear):
                x = l(x, rng=rng, mode=mode)
            else:
                x = l(x)
        return x

class MLP(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            input_dim: int = 1,
            hidden_dims: tuple =(64,),
            activation: str = 'silu',
            output_dim: int = 1,
            normalize: str = 'false',
            outact: str = 'none',
            dropout: float = 0.,
            **kwargs
        ):
        mlp_layers = (input_dim, *hidden_dims)
        layers = []
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:]):
            layers.append(nnx.Linear(in_dim, out_dim, rngs=rngs))
            if dropout > 0:
                layers.append(nnx.Dropout(dropout, rngs=rngs))
            if normalize != 'false':
                layers.append(NORMS[normalize](out_dim, rngs=rngs))
            layers.append(ACTIVATIONS[activation])
        if kwargs.get('zero_out_init', False):
            print('Initializing last layer to 0')
            layers.append(nnx.Linear(mlp_layers[-1], output_dim, rngs=rngs, kernel_init=initializers.zeros_init()))
        elif kwargs.get('bias_init', 0) > 0:
            bias_init = kwargs.get('bias_init', 0)
            print(f'Initializing last layer bias to {bias_init}')
            layers.append(nnx.Linear(mlp_layers[-1], output_dim, rngs=rngs, bias_init=initializers.constant(bias_init)))
        else:
            layers.append(nnx.Linear(mlp_layers[-1], output_dim, rngs=rngs))
        layers.append(ACTIVATIONS[outact])
        # self.layers = layers
        self.layers = nnx.List(layers)

    def __call__(self, x : jnp.ndarray):
        for l in self.layers:
            x = l(x)
        return x


def get_out_shape(in_shape, kernel_size, stride, transposed):
    height, width, channels = in_shape
    kernel_h, kernel_w = (kernel_size, kernel_size )if isinstance(kernel_size, int) else kernel_size
    new_width = int((width-kernel_w) / stride + 1) if not transposed else int((width-1)*stride + kernel_w)
    new_height = int((height-kernel_h) / stride + 1) if not transposed else int((height-1)*stride + kernel_h)

    return new_height, new_width


class ResidualBlock(nnx.Module):
    def __init__(self,
                 rngs: nnx.Rngs,
                 input_shape: tuple = (32,32,1),
                 out_channels: int = 1,
                 cnn_blocks : int = 2,
                 activation: str = 'silu',
                 transposed: bool = False,
                 kernel_size: tuple = (3,3),
                 stride: int = 2,
                 dropout: float = 0.
        ):

        ConvFn = nnx.ConvTranspose if transposed else nnx.Conv
        self.input_shape = input_shape
        self.output_shape = (*get_out_shape(input_shape, kernel_size, stride, transposed), out_channels)
        preconv_layers = [ConvFn(input_shape[-1], out_channels, kernel_size=kernel_size, strides=stride, padding='VALID', rngs=rngs), nnx.RMSNorm(out_channels, rngs=rngs)]
        if dropout > 0.:
            preconv_layers.append(nnx.Dropout(dropout, rngs=rngs))
        preconv_layers.append(ACTIVATIONS[activation])
        self.preconv = nnx.Sequential(*preconv_layers)
        # cnn_blocks
        layers = []
        for _ in range(cnn_blocks):
            layers = [nnx.RMSNorm(out_channels, rngs=rngs), ACTIVATIONS[activation], ConvFn(out_channels, out_channels, kernel_size=(1,1), strides=1, rngs=rngs)]
            if dropout:
                layers.append(nnx.Dropout(dropout, rngs=rngs))
        self.residual_conv = nnx.Sequential(*layers)

    def __call__(self, x):
        preconv_x = self.preconv(x)
        x = self.residual_conv(preconv_x)
        return preconv_x + x



class ResidualEncoder(nnx.Module):
    def __init__(self, 
                 rngs: nnx.Rngs, 
                 input_shape: tuple = (32,32,1),
                 cnn_blocks : int = 2,
                 depth : int = 24,
                 kernel_size: tuple = (3,3),
                 stride: int=2,
                 min_resolution : int = 4,
                 cnn_activation : str = 'silu',
                 mlp_layers : tuple = (5,),
                 mlp_activation : str = 'silu',
                 dropout : float = 0.,
                 final_activation='tanh',
                 pos_embed: bool = True,
                 **kwargs):
        
        n_res_layers = int(np.log2(input_shape[-2]) - np.log2(min_resolution))
        self.pos_embed = pos_embed
        if pos_embed:
            x_pos, y_pos = jnp.linspace(0, 1, input_shape[-3]), jnp.linspace(0, 1, input_shape[-2])
            self.x_pos, self.y_pos = jnp.meshgrid(x_pos, y_pos)
            input_shape = (*input_shape[:-1], input_shape[-1]+2)
        _in_shape = input_shape
        residual_blocks = []
        for i in range(n_res_layers):
            residual_blocks.append(ResidualBlock(rngs=rngs,
                                                 input_shape=_in_shape,
                                                 out_channels=depth*(2**i),
                                                 cnn_blocks=cnn_blocks,
                                                 activation=cnn_activation,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 transposed=False,
                                                 dropout=dropout))
            _in_shape = residual_blocks[-1].output_shape
        
        self.residual_stack = nnx.Sequential(*residual_blocks)

        input_dim = prod(residual_blocks[-1].output_shape)
        self.mlp = MLP(
            rngs=rngs,
            input_dim=input_dim,
            hidden_dims=mlp_layers[:-1],
            activation=mlp_activation,
            output_dim=mlp_layers[-1],
            outact=final_activation,
            normalize='rms'
        )
        

    def __call__(self, x):
        batch_dims = x.shape[:-3]
        bsize = prod(batch_dims)
        if self.pos_embed:
            x_pos = self.x_pos[None].repeat(bsize, axis=0).reshape(*x.shape[:-1], 1)
            y_pos = self.y_pos[None].repeat(bsize, axis=0).reshape(*x.shape[:-1], 1)
            x = jnp.concatenate([x, x_pos, y_pos], axis=-1)
        x = self.residual_stack(x)
        x = x.reshape(*x.shape[:-3], -1) # flatten
        x = self.mlp(x)
        return x

class ResidualDecoder(nnx.Module):

    def __init__(self,
                rngs : nnx.Rngs,
                input_dim : int,
                output_shape: tuple = (32, 32, 3),
                depth : int = 24,
                min_resolution : int = 4,
                cnn_activation : str = 'silu',
                cnn_blocks: int = 2,
                kernel_size : tuple = (3,3),
                stride : int = 2,
                mlp_layers : tuple = (5,),
                mlp_activation : str = 'silu',
                dropout : float = 0.,
                final_activation : str = 'tanh',
                **kwargs
        ):
        self.min_resolution = min_resolution
        self.output_shape = output_shape
        self.input_dim = input_dim
        self.depth=depth
        self.mlp_layers = [self.input_dim,] + mlp_layers
        n_res_layers_1 = int(np.ceil(np.log2(self.output_shape[0]) - np.log2(self.min_resolution)))
        n_res_layers_2 = int(np.ceil(np.log2(self.output_shape[1]) - np.log2(self.min_resolution)))
        n_res_layers = max(n_res_layers_1, n_res_layers_2)
        self.final_depth = int((2**(n_res_layers-1)) * self.depth)
        up_projection_dim = self.min_resolution**2 * self.final_depth
        self.mlp = MLP(rngs=rngs, input_dim=mlp_layers[0], hidden_dims=mlp_layers[1:], output_dim=up_projection_dim, activation=mlp_activation, normalize='rms')
        residual_blocks = []
        _in_shape = (self.min_resolution, self.min_resolution, self.final_depth)
        
        for i in reversed(range(n_res_layers)):
            residual_blocks.append(ResidualBlock(
                rngs=rngs,
                input_shape=_in_shape,
                out_channels=depth*(2**(i-1)) if i > 0 else output_shape[-1],
                cnn_blocks=cnn_blocks,
                kernel_size=kernel_size,
                stride=stride,
                activation=cnn_activation,
                dropout=dropout,
                transposed=True
            ))
            _in_shape = residual_blocks[-1].output_shape
        self.residual_stack = nnx.Sequential(*residual_blocks)
        output_shape = residual_blocks[-1].output_shape
        out_height, out_width = output_shape[-3:-1]
        center_h, center_w = out_height // 2, out_width // 2
        self.out_shape_h, self.out_shape_w = self.output_shape[:-1]
        self.crop_start_h = center_h - self.out_shape_h // 2
        self.crop_start_w = center_w - self.out_shape_w // 2
        self.final_activation = ACTIVATIONS[final_activation]


    def __call__(self, x):
        batch_dims = x.shape[:-1]
        x = self.mlp(x) # up project
        x = x.reshape(-1, self.min_resolution, self.min_resolution, self.final_depth)
        x = self.residual_stack(x)

        # crop
        x = jax.lax.slice_in_dim(x, self.crop_start_h, self.crop_start_h+self.out_shape_h, axis=-3)
        x = jax.lax.slice_in_dim(x, self.crop_start_w, self.crop_start_w+self.out_shape_w, axis=-2)
        x = self.final_activation(x)
        return x.reshape(*batch_dims, *x.shape[1:])

def build_multihead_attention(rngs, **kwargs):
    del kwargs['type']
    return nnx.MultiHeadAttention(**kwargs, rngs=rngs)

MODEL_FACTORIES = {
    'residual_encoder': ResidualEncoder,
    'residual_decoder': ResidualDecoder,
    'mlp': MLP,
    'noisy_mlp': NoisyMLP,
    'blockgru' : BlockGRUCell,
    'gru' : GRUCell,
    'multihead_attention': build_multihead_attention
}

def build_model(config, rngs : nnx.Rngs, id=None):
    model =  MODEL_FACTORIES[config.type]
    return model(rngs=rngs, **config)