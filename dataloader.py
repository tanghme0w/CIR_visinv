from torch.utils.data import DataLoader


class TripletDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
