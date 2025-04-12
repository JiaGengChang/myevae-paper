# dictionary that allows object-like dot accessing of its keys
class AttrDict(dict):
    """
    obj = AttrDict()
    obj.somefield = "somevalue"
    > obj.somefield
    > somevalue
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value