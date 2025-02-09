import json
import nltk

nltk.download('punkt_tab')
dataset_names = {
    'TASK1': ['twitter19', 'weibo19'],
}


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json'))
            wiki = json.load(open(f'../data/proxy/entity_from_wiki/{task}_{dataset}.json'))

            save_dir = f'../data/proxy/retrieval/{task}_{dataset}.json'
            out = []

            fail_ids = []
            for idx in range(len(data)):
                item, each_wiki = data[idx], wiki[idx]
                # print(each_wiki)
                out_text = item[0]
                print(out_text)
                try: 
                    for entity, exp in each_wiki:
                        if exp is None:
                            continue
                        exp = ' '.join(nltk.sent_tokenize(exp)[:3])
                        entity_index = out_text.lower().find(entity.lower())
                        if entity_index != -1:
                            entity_index = entity_index + len(entity)
                            out_text = out_text[:entity_index] + ' (' + exp + ')' + out_text[entity_index:]
                    out.append(out_text)
                except Exception as e:
                    out.append("")
                    # print(each_wiki)
                    fail_ids.append(idx)
            print(fail_ids)
            json.dump(out, open(save_dir, 'w'))


if __name__ == '__main__':
    main()
