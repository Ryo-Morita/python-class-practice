from dataclasses import dataclass, field

@dataclass
class Mydata():
    name: str = 'karaage'
    option: list = field(default_factory=list)