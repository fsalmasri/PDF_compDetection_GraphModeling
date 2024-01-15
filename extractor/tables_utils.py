from . import doc


def delete_update_tables(to_delete_lst, ref='paths'):
    sp = doc.get_current_page()

    if ref == 'paths':
        # Clean paths list
        to_keep_paths = {k_path: v_path for k_path, v_path in sp.paths_lst.items() if k_path not in to_delete_lst}
        to_delete_paths = {k_path: v_path for k_path, v_path in sp.paths_lst.items() if k_path in to_delete_lst}
        print(f'duplicates paths to delete {len(to_delete_paths)}, Remaining {len(to_keep_paths)}')

        sp.paths_lst = to_keep_paths.copy()

        # Clean nodes list
        to_delete_edges = [tuple([x['p1'], x['p2']]) for x in to_delete_paths.values()]
        to_delete_nodes = {item for tpl in to_delete_edges for item in tpl}
        to_keep_nodes = set([coord for x in to_keep_paths.values() for coord in [x['p1'], x['p2']]])
        remaining_nodes = list(to_delete_nodes - to_keep_nodes)

        if remaining_nodes:
            filtered_nodes = list(to_keep_nodes - to_delete_nodes)
            # //TODO delete them from node list
            # //TODO Clean Primitives


            raise NotImplementedError
        else:
            print(f'no nodes to delete')

        # Clean Graph
        sp.G.remove_edges_from(to_delete_edges)

        sp.save_data(61)


    else:
        raise NotImplementedError

def clean_duplicates_paths():
    sp = doc.get_current_page()

    seen_paths = set()
    to_delete_lst = []

    selected_paths = {k:v for k, v in sp.paths_lst.items() if v["item_type"] == "l" and v["path_type"] == "f"}

    for k_path, v_path in selected_paths.items():
        path_tuple = tuple((tuple(sp.nodes_LUT[v_path['p1']]), tuple(sp.nodes_LUT[v_path['p2']])))

        if path_tuple in seen_paths:
            to_delete_lst.append(k_path)
        else:
            seen_paths.add(path_tuple)

    if to_delete_lst:
        delete_update_tables(to_delete_lst, ref='paths')