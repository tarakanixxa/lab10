import json,re,argparse
from pathlib import Path
def load(p): return [json.loads(x) for x in open(p,encoding="utf8") if x.strip()]
def save(rows,p):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with open(p,"w",encoding="utf8") as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
def pack(d,text,i,s):
 return {**{k:d[k] for k in ["doc_id","title","section","type","date","sensitivity","source_url"]},"chunk_id":f"{d['doc_id']}_{s}_{i:03d}","text":text,"strategy":s}
def fixed(ds,size=260,overlap=40):
 out=[]
 for d in ds:
  t=re.sub(r"\s+"," ",d["text"]).strip(); start=0; i=1
  while start<len(t):
   end=min(len(t),start+size); out.append(pack(d,t[start:end],i,"fixed"))
   if end==len(t):break
   start=end-overlap;i+=1
 return out
def structure(ds): return [pack(d,re.sub(r"\s+"," ",d["text"]).strip(),i,"structure") for i,d in enumerate(ds,1)]
def semantic(ds,size=420):
 out=[]
 for d in ds:
  ss=re.split(r"(?<=[.!?])\s+",d["text"].strip()); buf=[]; n=0; i=1
  for s in ss:
   if buf and n+len(s)>size:out.append(pack(d," ".join(buf),i,"semantic"));i+=1;buf=[];n=0
   buf.append(s);n+=len(s)+1
  if buf:out.append(pack(d," ".join(buf),i,"semantic"))
 return out
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--input",default="data/corpus.jsonl");p.add_argument("--out",default="data/chunks");a=p.parse_args();ds=load(a.input)
 for name,fn in [("fixed",fixed),("structure",structure),("semantic",semantic)]:save(fn(ds),f"{a.out}/{name}.jsonl")
 print("Chunking complete")
