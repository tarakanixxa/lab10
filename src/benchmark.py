import json,argparse,time
from index import load
from hybrid import BM25,dense,rrf,rerank
from evaluate import recall,mrr,ndcg
def run(corpus,gold,key):
 rows=load(corpus);qs=[json.loads(x) for x in open(gold,encoding="utf8") if x.strip()];bm=BM25(rows);out={k:[] for k in ["BM25","Dense","Hybrid","Hybrid+rerank"]}
 for q in qs:
  b=[rows[i]["chunk_id"] for i in bm.search(q["query"],10)];dids=dense(q["query"],rows,key,k=30);d=[rows[i]["chunk_id"] for i in dids[:10]];f=rrf([bm.search(q["query"],30),dids],60,30);h=[rows[i]["chunk_id"] for i in f[:10]]
  try:rr=[rows[i]["chunk_id"] for i in rerank(q["query"],rows,f,10)]
  except Exception:rr=h
  for name,r in zip(out,[b,d,h,rr]):out[name].append(r)
 for name,rankings in out.items():
  vals=[(recall(r,q["relevant_chunk_ids"],5),recall(r,q["relevant_chunk_ids"],10),mrr(r,q["relevant_chunk_ids"]),ndcg(r,q["relevant_chunk_ids"])) for r,q in zip(rankings,qs)];print(name,[round(sum(v[i] for v in vals)/len(vals),4) for i in range(4)])
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--corpus",default="data/corpus.jsonl");p.add_argument("--gold",default="eval/gold.jsonl");p.add_argument("--model",choices=["e5","minilm"],default="e5");a=p.parse_args();run(a.corpus,a.gold,a.model)
