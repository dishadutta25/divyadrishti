"""Dataset validation summary and prediction diagnostics; no fabricated scores."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from visionx.data import JsonlDetectionDataset
from visionx.model import GridDetector
from visionx.model import decode


def box_iou(a, b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[0]+a[2],b[0]+b[2]),min(a[1]+a[3],b[1]+b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1e-9,a[2]*a[3]+b[2]*b[3]-inter)


def average_precision(predictions, ground_truth, class_id, threshold=.5):
    gt={idx:[b for cid,b in boxes if cid==class_id] for idx,boxes in ground_truth.items()}
    positives=sum(len(v) for v in gt.values())
    if positives == 0: return None
    matched={idx:set() for idx in gt}; ranked=sorted((p for p in predictions if p[1]==class_id),key=lambda p:p[2],reverse=True)
    tp=[]; fp=[]
    for idx,_,_,box in ranked:
        candidates=gt.get(idx,[])
        overlaps=[(box_iou(box,b),j) for j,b in enumerate(candidates) if j not in matched.get(idx,set())]
        best=max(overlaps,default=(0,-1))
        ok=best[0]>=threshold
        tp.append(1 if ok else 0); fp.append(0 if ok else 1)
        if ok: matched[idx].add(best[1])
    if not ranked: return 0.0
    cum_tp=torch.tensor(tp,dtype=torch.float32).cumsum(0).tolist(); cum_fp=torch.tensor(fp,dtype=torch.float32).cumsum(0).tolist()
    recalls=[x/positives for x in cum_tp]; precisions=[t/max(1e-9,t+f) for t,f in zip(cum_tp,cum_fp)]
    return sum(max((p for r,p in zip(recalls,precisions) if r>=level),default=0.0) for level in [i/100 for i in range(101)])/101


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--annotations",required=True); ap.add_argument("--classes",required=True); ap.add_argument("--device",default="cpu"); ap.add_argument("--confidence",type=float,default=.35)
    a=ap.parse_args(); names=[x.strip() for x in Path(a.classes).read_text(encoding="utf-8").splitlines() if x.strip()]
    ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if ck["class_names"] != names: ap.error("checkpoint and classes file differ")
    ds=JsonlDetectionDataset(a.annotations,names,ck["image_size"],ck["grid_size"])
    model=GridDetector(len(names),ck["grid_size"]).to(a.device); model.load_state_dict(ck["model"]); model.eval()
    seen=positive=predicted=tp_at_conf=0; predictions=[]; ground_truth={}
    with torch.no_grad():
        for idx,(images, _) in enumerate(DataLoader(ds,batch_size=1)):
            rec=ds.records[idx]; iw,ih=int(rec.get("width",ck["image_size"])),int(rec.get("height",ck["image_size"]))
            gt=[]
            for b in rec.get("boxes",[]):
                gt.append((int(b["class_id"]),(float(b["x"]),float(b["y"]),float(b["width"]),float(b["height"]))))
            ground_truth[idx]=gt; positive+=len(gt); seen+=1
            out=model(images.to(a.device))[0].cpu()
            rows=decode(out,names,ck["image_size"],ck["image_size"],0.001)
            for cid,score,x,y,w,h in rows:
                # Dataset preprocessing stretches images to a square; invert that scaling.
                box=(x*iw/ck["image_size"],y*ih/ck["image_size"],w*iw/ck["image_size"],h*ih/ck["image_size"])
                predictions.append((idx,cid,score,box))
                if score>=a.confidence:
                    predicted+=1
    used=set();
    for idx,cid,score,box in sorted((p for p in predictions if p[2]>=a.confidence),key=lambda p:p[2],reverse=True):
        candidates=[(box_iou(box,b),j) for j,(c,b) in enumerate(ground_truth[idx]) if c==cid and (idx,cid,j) not in used]
        best=max(candidates,default=(0,-1))
        if best[0]>=.5: tp_at_conf+=1; used.add((idx,cid,best[1]))
    aps=[average_precision(predictions,ground_truth,c) for c in range(len(names))]
    aps=[v for v in aps if v is not None]
    precision=tp_at_conf/max(1,predicted); recall=tp_at_conf/max(1,positive)
    print(f"records={seen} labeled_objects={positive} predictions_at_confidence={predicted}")
    print(f"precision@{a.confidence:.2f},IoU0.50={precision:.4f} recall@{a.confidence:.2f},IoU0.50={recall:.4f}")
    print(f"mAP50_101point={sum(aps)/len(aps):.4f}" if aps else "mAP50_101point=n/a (no labeled classes in this split)")

if __name__ == "__main__": main()
