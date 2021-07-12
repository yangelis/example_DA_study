import config
import yaml

with open('config.yaml', 'w') as file:
    documents = yaml.dump({'configuration': config.configuration}, file)
