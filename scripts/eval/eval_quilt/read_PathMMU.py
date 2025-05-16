import json


path = "/ssd1/trinh/data/ViLa/images/PathAsst/PathMMU/"
folders = ['images', 'data.json', '.gitattributes', 'images.zip', '.git', 'README.md']

with open(f'{path}/data.json') as f:
    data = json.load(f)

print(len(data))
# dict_keys(['PubMed', 'SocialPath', 'EduContent', 'PathCLS'])
