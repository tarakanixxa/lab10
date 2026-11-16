import re,math,argparse,json
from collections import Counter,defaultdict
from index import load,embed,MODELS
def tok(s):return re.findall(r"[0-9a-zа-яіїєґ]+",s.lower())
class BM25:
 def __init__(self,rows):
  self.rows=rows;self.ds=[tok(x["text"]) for x in rows];self.avg=sum(map(len,self.ds))/max(1,len(rows));self.df=Counter()
  for d in self.ds:self.df.update(set(d))
 def search(self,q,k=30,allowed=None):
  scores=[]
  for i,d in enumerate(self.ds):
   if allowed is not None and i not in allowed:continue
   tf=Counter(d);score=0
   for w in tok(q):
    if tf[w]:
     idf=math.log(1+(len(self.ds)-self.df[w]+.5)/(self.df[w]+.5));score+=idf*tf[w]*2.5/(tf[w]+1.5*(.25+.75*len(d)/(self.avg or 1)))
   scores.append((i,score))
  return [i for i,s in sorted(scores,key=lambda x:x[1],reverse=True)[:k]]
def prefilter(rows,type_=None,sensitivity=None,date_from=None,date_to=None):
 return {i for i,r in enumerate(rows) if (not type_ or r["type"]==type_) and (not sensitivity or r["sensitivity"]==sensitivity) and (not date_from or r["date"]>=date_from) and (not date_to or r["date"]<=date_to)}
def dense(q,rows,key="e5",allowed=None,k=30):
 name=MODELS[key];v=embed([r["text"] for r in rows],name);qv=embed([q],name,True)[0];ids=list(range(len(rows))) if allowed is None else list(allowed);return sorted(ids,key=lambda i:float(v[i]@qv),reverse=True)[:k]
def rrf(lists,k0=60,n=30):
 scores=defaultdict(float)
 for ls in lists:
  for rank,i in enumerate(ls,1):scores[i]+=1/(k0+rank)
 return [i for i,s in sorted(scores.items(),key=lambda x:x[1],reverse=True)[:n]]
def rerank(q,rows,ids,n=5):
 from sentence_transformers import CrossEncoder
 m=CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1");scores=m.predict([(q,rows[i]["text"]) for i in ids]);return [i for i,s in sorted(zip(ids,scores),key=lambda x:float(x[1]),reverse=True)[:n]]
def search(q,rows,key="e5",filters=None,use_rerank=True):
 allowed=prefilter(rows,**(filters or {}));b=BM25(rows).search(q,30,allowed);d=dense(q,rows,key,allowed,30);f=rrf([b,d],60,30)
 try:final=rerank(q,rows,f,5) if use_rerank else f[:5]
 except Exception:final=f[:5]
 return {"bm25":b,"dense":d,"rrf":f,"final":final}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--query",required=True);p.add_argument("--input",default="data/corpus.jsonl");p.add_argument("--no-rerank",action="store_true");a=p.parse_args();rs=load(a.input);z=search(a.query,rs,use_rerank=not a.no_rerank);[print(rs[i]["chunk_id"],rs[i]["text"]) for i in z["final"]]
