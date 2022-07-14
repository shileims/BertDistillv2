import inspect

class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()


    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, cls):
        if not inspect.isclass(cls):
            raise TypeError('module must be a class, but got {}'.format(
                type(cls)))
        name = cls.__name__
        if name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                name, self.name))
        self._module_dict[name] = cls

    def register_module(self, cls):
        self._register_module(cls)
        return cls
