import os
import pandas as pd

deg_result_path = "/home/hb/python/lof/deg/norman"
print('현재까지 계산 완료된 DEG 데이터 개수:', len([x for x in os.listdir(deg_result_path) if '.csv' in x]))

import pandas as pd
import os
import requests
import json
os.chdir("/home/hb/python/lof/deg/norman")
CONDITION = 'TGFBR2_IGDCC3'
path = '/home/hb/python/lof/benchmark/sigCOM_lincs'

def deg_analysis(CONDITION):
    deg = pd.read_csv(f"{CONDITION}.csv")
    try: 
        deg = deg.drop(['Unnamed: 0'], axis=1)
    except:
        pass
    column = deg.columns[1]
    deg_up = deg.sort_values(f"{column}", ascending=False).dropna(axis=0).set_index("gene_name")
    deg_down = deg.sort_values(f"{column}", ascending=True).dropna(axis=0).set_index("gene_name")
    up_genes = list(deg_up.iloc[:600].index.values)
    down_genes = list(deg_down.iloc[:600].index.values)

    METADATA_API = "https://maayanlab.cloud/sigcom-lincs/metadata-api/"
    DATA_API = "https://maayanlab.cloud/sigcom-lincs/data-api/api/v1/"

    input_gene_set = {
        "up_genes": up_genes,
        "down_genes": down_genes
    }

    all_genes = input_gene_set["up_genes"] + input_gene_set["down_genes"]

    payload = {
        "filter": {
            "where": {
                "meta.symbol": {
                    "inq": all_genes
                }
            },
            "fields": ["id", "meta.symbol"]
        }
    }
    res = requests.post(METADATA_API + "entities/find", json=payload)
    entities = res.json()

    for_enrichment = {
        "up_entities": [],
        "down_entities": []
    }

    for e in entities:
        symbol = e["meta"]["symbol"]
        if symbol in input_gene_set["up_genes"]:
            for_enrichment["up_entities"].append(e["id"])
        elif symbol in input_gene_set["down_genes"]:
            for_enrichment["down_entities"].append(e["id"])
    for_enrichment['up_entities'] = for_enrichment['up_entities'][:500]
    for_enrichment['down_entities'] = for_enrichment['down_entities'][:500]
    print(len(for_enrichment['up_entities']), len(for_enrichment['down_entities']))

    payload = {
        "meta": {
            "$validator": "/dcic/signature-commons-schema/v6/meta/user_input/user_input.json",
            **for_enrichment
        },
        "type": "signature"
    }
    res = requests.post(METADATA_API + "user_input", json=payload)
    persistent_id = res.json()["id"]
    print("Access your analysis here: https://maayanlab.cloud/sigcom-lincs#/SignatureSearch/Rank/%s"%persistent_id)

    NUM_CASE = 120
    query = {
        **for_enrichment,
        "limit": NUM_CASE,
        "database": "l1000_xpr"
    }

    res = requests.post(DATA_API + "enrich/ranktwosided", json=query)
    results = res.json()

    # Optional, multiply z-down and direction-down with -1
    for i in results["results"]:
        i["z-down"] = -i["z-down"]
        i["direction-down"] = -i["direction-down"]

    results = results['results'][:NUM_CASE]
    sigids = {i['uuid']: i for i in results}

    payload = {
        "filter": {
            "where": {
                "id": {
                    "inq": list(sigids.keys())
                }
            }
        }
    }

    res = requests.post(METADATA_API + "signatures/find", json=payload)
    signatures = res.json()

    ## Merge the scores and the metadata
    for sig in signatures:
        uid = sig["id"]
        scores = sigids[uid]
        # scores.pop("uuid")
        sig["scores"] = scores

    for i in signatures:
        dict1 = i['meta']
        dict2 = i['scores']

        dict1.update(dict2)

    df_mimickers = pd.DataFrame([x['meta'] for x in signatures])
    file_name = f'Norman_K562_{CONDITION}'

    df_mimickers = df_mimickers.sort_values(by=['z-sum'], ascending=False)
    df_mimickers.index = [f'Top-{x}' for x in range(1, len(df_mimickers)+1)]
    df_mimickers.dropna(subset=['pert_name'], inplace=True)
    # df_mimickers.dropna(subset=['pert_name'], ).resbet_index(drop=True, )
    # df_mimickers.dropna(subset=['pert_name'], inplace=True)
    # df_mimickers['z-sum']
    mimickers_save_path = f"{path}/{file_name}.csv"
    df_mimickers.to_csv(mimickers_save_path)

    df_mimickers_top10 = df_mimickers.iloc[:10]
    df_mimickers_top30 = df_mimickers.iloc[:30]
    df_mimickers_top50 = df_mimickers.iloc[:50]
    df_mimickers_top100 = df_mimickers.iloc[:100]

    benchmark = {}

    for i, rank in zip([df_mimickers_top10, df_mimickers_top30, df_mimickers_top50, df_mimickers_top100], [10, 30, 50, 100]):
        pert_list = i.pert_name.unique()
        if CONDITION not in benchmark.keys():
            benchmark[CONDITION] = {}
        if CONDITION in pert_list:
            print(f'TOP-{rank} Accuracy: True')
            benchmark[CONDITION][f'Top-{rank}'] = 1
        else: 
            print(f'Top-{rank} Accuracy: False')
            benchmark[CONDITION][f'Top-{rank}'] = 0
    new_benchmark_df = pd.DataFrame(benchmark)
    benchmark_df = pd.read_csv("/home/hb/python/lof/benchmark/sigCOM_lincs/benchmark_Norman.csv", index_col=0)
    if CONDITION not in benchmark_df.columns:
        benchmark_df = pd.concat([benchmark_df, new_benchmark_df], axis=1)
        benchmark_df.to_csv("/home/hb/python/lof/benchmark/sigCOM_lincs/benchmark_Norman.csv", index=True)
    else:
        print('already exists')
    # mimickers = [x for x in results['results'] if x['type']=='mimickers']
    return benchmark_df