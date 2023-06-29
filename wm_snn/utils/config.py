from typing import Any


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __repr__(self) -> str:
        return self.__dict__.__repr__()
    