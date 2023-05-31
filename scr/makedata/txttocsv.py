# with open('../../data/dataset.txt', 'r', encoding='utf-8') as f1, open('../../data/output1.txt', 'w', encoding='utf-8') as f2:
#     for line in f1:
#         modified_line = line.replace('|', ' |')
#         f2.write(modified_line)
#
# with open('../../data/output1.txt', 'r', encoding='utf-8') as f1, open('../../data/output2.txt', 'w', encoding='utf-8') as f2:
#     for line in f1:
#         modified_line = line.replace('  ', ' ')
#         f2.write(modified_line)

import csv

with open('D:/study/code/Source/pycharm_workspace/abuse_division/data/dataset.txt', 'r', encoding='utf-8') as f1, open('D:/study/code/Source/pycharm_workspace/abuse_division/data/origin_final.csv', 'w', newline='', encoding='utf-8') as f2:
    writer = csv.writer(f2)
    writer.writerow(['text', 'label'])
    try:
        for line in f1:
            parts = line.strip().split('|')
            text = parts[0].strip()
            label = parts[1].strip()

            writer.writerow([text, label])
    except IndexError:
        print('error')