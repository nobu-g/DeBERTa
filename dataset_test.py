import random

from torch.utils.data import DataLoader, Dataset


class DynamicDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        initial_buff_size = 50
        self.data: list[int] = list(range(initial_buff_size))
        self.offset = 0
        random.seed(0)

    def __getitem__(self, index: int):
        if index - self.offset >= len(self.data):
            buff_size = random.randint(5, 10) * 10
            self.offset += len(self.data)
            self.data = [i for i in range(self.offset, self.offset + buff_size)]
            print("data extended, buff size:", buff_size)
        # return (index, self.offset, self.data[index - self.offset])
        return self.data[index - self.offset]

    def __len__(self) -> int:
        return 500


def main():
    dataset = DynamicDataset()
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=1)
    for batch in dataloader:
        print(batch)


if __name__ == "__main__":
    main()
