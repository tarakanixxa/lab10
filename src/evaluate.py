import json,math,argparse
def recall(r,rel,k):
 rel=set(rel);return (1.0 if not r else 0.0) if not rel else len(set(r[:k])&rel)/len(rel)
def mrr(r,rel):
 rel=set(rel);return next((1/i for i,x in enumerate(r,1) if x in rel),0.0)
def ndcg(r,rel,k=10):
 rel=set(rel);dcg=sum(1/math.log2(i+2) for i,x in enumerate(r[:k]) if x in rel);ideal=sum(1/math.log2(i+2) for i in range(min(k,len(rel))));return dcg/ideal if ideal else 0.0
def kappa(a,b):
 assert len(a)==len(b) and a;labs=set(a)|set(b);n=len(a);po=sum(x==y for x,y in zip(a,b))/n;pe=sum(sum(x==l for x in a)/n*sum(x==l for x in b)/n for l in labs);return (po-pe)/(1-pe) if pe<1 else 1.0
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--gold",default="eval/gold.jsonl");a=p.parse_args();g=[json.loads(x) for x in open(a.gold,encoding="utf8") if x.strip()];print("queries",len(g),"unanswerable",sum(not x["answerable"] for x in g));print("kappa demonstration",kappa([bool(x["annotator_1"]) for x in g],[bool(x["annotator_2"]) for x in g]))
