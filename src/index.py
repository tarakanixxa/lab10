import json,time,argparse,pickle
from pathlib import Path
import numpy as np
MODELS={"e5":"intfloat/multilingual-e5-small","minilm":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
def load(p):return [json.loads(x) for x in open(p,encoding="utf8") if x.strip()]
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=2)
def get_model(name):
    return SentenceTransformer(name, device="cpu")

def embed(texts, name, is_query=False):
    model = get_model(name)

    prefix = ("query: " if is_query else "passage: ") if "e5" in name else ""
    texts = [prefix + text for text in texts]

    return model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True
    )
def build(src,out,key="e5",M=32,efSearch=64):
 import faiss
 rows=load(src);name=MODELS[key];t=time.perf_counter();v=embed([r["text"] for r in rows],name);elapsed=time.perf_counter()-t
 ix=faiss.IndexHNSWFlat(v.shape[1],M,faiss.METRIC_INNER_PRODUCT);ix.hnsw.efConstruction=80;ix.hnsw.efSearch=efSearch;ix.add(np.asarray(v,dtype="float32"))
 Path(out).mkdir(parents=True,exist_ok=True);faiss.write_index(ix,f"{out}/{key}.faiss");pickle.dump(rows,open(f"{out}/{key}.pkl","wb"))
 json.dump({"model":name,"M":M,"efConstruction":80,"efSearch":efSearch,"build_seconds":elapsed,"vectors":len(rows),"dimension":int(v.shape[1])},open(f"{out}/{key}.params.json","w"),indent=2)
 return elapsed
def search(q,idx,meta,key="e5",k=5):
 import faiss
 ix=faiss.read_index(str(idx));rows=pickle.load(open(meta,"rb"));vec=embed([q],MODELS[key],True).astype("float32");t=time.perf_counter();s,ids=ix.search(vec,k);return [(rows[i],float(v)) for i,v in zip(ids[0],s[0]) if i>=0],time.perf_counter()-t
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",default="data/corpus.jsonl");p.add_argument("--out",default="artifacts/index");p.add_argument("--model",choices=MODELS,default="e5");a=p.parse_args();print("Build seconds:",build(a.input,a.out,a.model))
