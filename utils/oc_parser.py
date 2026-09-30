
from omegaconf import OmegaConf as oc
import re

def parse_oc_args(oc_args):
    assert len(oc_args)%2==0, f'args with len {len(oc_args)}: {oc_args}'
    oc_args = ['='.join([oc_args[i].split('--')[-1], oc_args[i+1]]) for i in range(len(oc_args)) if i%2==0]
    cli_config = oc.from_cli(oc_args)
    return cli_config

def safe_merge(config, cli_config):
    pass

def parse_tuning_args(args):
    '''
        parse args of the form --param.{name} value
        return: dict[param_name, values]
    '''
    params = {}
    pattern = '--param.([a-z_]+)'
    args = list(reversed(args))

    unknown_args = []
    while len(args) > 0:
        e = args.pop()
        match = re.match(pattern, e)
        if match:
            param_name = match.group(1)
            params[param_name] = [] 
            while len(args) > 0:
                e = args.pop()
                match = re.match(pattern, e)
                if not match and not (e.startswith('--') or e.startswith('-')):
                    params[param_name].append(e)
                else:
                    args.append(e)
                    break
        elif e.startswith('--') or e.startswith('-'):
            # unknown arg
            unknown_args.append(e)
            while len(args) > 0:
                e = args.pop()
                if not e.startswith('--'):
                    unknown_args.append(e)
                else:
                    args.append(e)
                    break
    return params, unknown_args