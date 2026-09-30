import os
import pathlib

import json

import wandb

import pandas as pd
import numpy as np

class Logger:
    def __init__(self, outdir, expname):
        self.outdir = pathlib.Path(outdir)
        self.expname = expname
        if not self.outdir.exists():
            print(f'{self.outdir} does not exists! Making dir')
            os.makedirs(self.outdir / expname)
        elif (self.outdir / expname).exists():
            print(f'Experiment {expname} exists! You may be overwriting old results')
        else:
            os.makedirs(self.outdir / expname)
            print(f'Logging at {self.outdir / expname}')

    def log_dict(self, logs, prefix, **kwargs):
        raise NotImplementedError


class JSONLogger(Logger):

    def __init__(self,
                 outdir,
                 expname,
                 filename='results.jsonl',
                 overwrite=True
                ):
        super().__init__(outdir, expname)
        self.filename = filename
        if (self.outdir / self.expname / filename).exists() and overwrite:
            print(f'{filename} exists! Overwriting!')
            f = open(self.outdir / self.expname / self.filename, 'w')
            f.close()

    def log_dict(self, logs, prefix='', batched=True, **kwargs):
        if isinstance(logs, dict):
            logs = [logs]
        with open(self.outdir / self.expname / self.filename, 'a') as f:
            
            if batched:
                for log in logs:
                    log = {f'{prefix}/{k}': v for k,v in log.items()}
                    df = pd.DataFrame(log)
                    df.to_json(f, orient='records', lines=True)
            else:
                def to_string(log):
                    log = {f'{prefix}/{k}': float(v) for k,v in log.items()}
                    log_string = json.dumps(log)
                    return log_string
                log_lines = '\n'.join([to_string(log) for log in logs]) + '\n'
                f.write(log_lines)


class WandBLogger(Logger):
    def __init__(
            self,
            outdir,
            expname,
            project,
            config={}
    ):
        super().__init__(outdir, expname)
        wandb.init(
            dir=self.outdir,
            project=project,
            name=expname,
            config=config
        )
    
    def log_dict(self, logs, prefix, step=None, **kwargs):
        if isinstance(logs, dict):
            logs = [logs]
        logs = [{f'{prefix}/{k}':v for k,v in log.items()} for log in logs]
        
        if isinstance(step, str):
            for log in logs:
                wandb.log(log, step=log[step])
        else:
            for log in logs:
                wandb.log(log)
    

class MultiLogger(Logger):

    def __init__(self, loggers):
        self.loggers = loggers

    def log_dict(self, logs, prefix, **kwargs):
        for logger in self.loggers:
            logger.log_dict(logs, prefix, **kwargs)