import json
import os
from tqdm import tqdm
from utils import get_response


dataset_names = {
    'TASK1': ['twitter19', 'weibo19'],

}


def get_one_response(data):
    father = data['father']
    res = data['res']
    response = []
    for index, item in enumerate(res):
        if father[index] is None:
            response.append(None)
        else:
            source = res[father[index]]
            target = res[index]
            out = get_response(source, target)
            response.append(out)
    assert len(response) == len(res)
    return response


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            network_dir = f'../data/networks/{task}_{dataset}'
            save_dir = f'../data/proxy/response/{task}_{dataset}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            files = os.listdir(network_dir)
            for file in tqdm(files):
                save_path = f'{save_dir}/{file}'
                if os.path.exists(save_path):
                    continue
                data = json.load(open(f'{network_dir}/{file}'))
                res = get_one_response(data)
                json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
