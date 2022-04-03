# -*- coding=utf-8 -*-

class _ImportRaiser(object):

    def __init__(self, name, command):
        self.name = name
        self.command = command

    def raise_import(self):
        msg = self.name + ' required. Try: ' + self.command
        raise ImportError(msg)

    def __getattr__(self, *args, **kwargs):
        self.raise_import()

    def __call__(self, *args, **kwargs):
        self.raise_import()
