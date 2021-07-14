# %%
"""
Example of a chronjob
"""

# %%
import tree_maker
from tree_maker import NodeJob


# %%
# Load the tree from a yaml
try:
    root=tree_maker.tree_from_json(
    f'./study_000/tree.json')
except Exception as e:
    print(e)
    print('Probably you forgot to edit the address of you json file...')

if root.has_been('completed'):
    print('All descendants of root are completed!')
else:
    for node in root.descendants:
        node.smart_run()
    if all([descendant.has_been('completed') for descendant in root.descendants]):
        root.tag_as('completed')
        print('All descendants of root are completed!')
