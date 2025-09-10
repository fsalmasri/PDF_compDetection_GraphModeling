from label_studio_sdk.client import LabelStudio
from label_studio_sdk import Client
from label_studio_sdk.core.api_error import ApiError

import requests

LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '1f62442ca2afb86b3ccccc5c7cf144d1cdceb13f'


client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

# client = LabelStudio(
#     api_key=API_KEY,
# )

print(ls.check_connection())
projects_list = client.projects.list()



# try:
#     for project in projects_list:
#         print(project)
# except ApiError as e:
#     print(e)


try:
    # project_info = client.tasks.create_many_status(
    #     id=4
    #
    # )
    # project_info = client.projects.import_tasks(
    #     id=1,
    #     request=[{}],
    # )
    project_info = client.projects.get(id=4)


    print(project_info)
    # for p in projects_list:
    #     print(p)

except ApiError as e:
    print(e)

try:
    # vs = client.views.list()
    # for v in vs:
    #     print(v)
    # exit()
    # annotated_tasks = client.tasks.list(project=4)




    tab = client.views.get(id=86)
    annotated_tasks = client.tasks.list(view=tab.id, fields='task_only')
    # print(len(annotated_tasks))
    for annotated_task in annotated_tasks:
        print(annotated_task)
        # print(annotated_task.id)
        # print(len(annotated_task.annotations))
        # print(annotated_task.storage_filename)
        # print(annotated_task.data['image'])

        # print(annotated_task.annotations[0]['task'])
        # print(annotated_task.annotations[0]['project'])
        # print(f'num of annots: {len(annotated_task.annotations[0]["result"])}')
        # for k in annotated_task.annotations[0]['result']:
        #     print(k['id'])
            # exit()

    exit()

    # for annotated_task in annotated_tasks:
    #     print(str(annotated_task.annotations[0].result[0]['value']['choices']))
    #     exit()

    # print(tasks)
    # print(tasks.get_next())
    # for task in tasks:
    #     print(task.id)
    #     print(task.storage_filename)
    #     print(task.data['image'])
    #     exit()

    # annots = client.annotations.list(id=13966)
    # annots = client.annotations.get( id=13716, )
    # annots = client.views.list(project=4)
    # print(annots)
    # for f in annots:
    #     print(f)
    #     exit()
    # print(annots)

#     annots = client.annotations.get(
#     id=446,
# )
#     print(annots)

except ApiError as e:
    print(e)




# print(project_info)