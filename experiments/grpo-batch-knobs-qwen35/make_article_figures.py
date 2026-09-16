"""Standalone plots and social card, built only from recorded measurements."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw, ImageFont

INK="#20252b"
ACCENT="#4f46e5"


def frame(title, mobile):
    fig,ax=plt.subplots(figsize=(4.0,3.5) if mobile else (7.2,4.1),layout="constrained")
    fig.patch.set_facecolor("#fbfbfa")
    ax.set_facecolor("#fbfbfa")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#cdd1d6")
    ax.tick_params(colors="#515963",labelsize=10 if mobile else 11)
    ax.set_title(title,loc="left",fontweight="bold",color=INK,pad=14,fontsize=13 if mobile else 15)
    ax.grid(axis="y",color="#e2e5e8",linewidth=.7)
    ax.set_axisbelow(True)
    return fig,ax


def save(fig,path,point_titles=None):
    fig.savefig(path,metadata={"Date":None},bbox_inches="tight")
    plt.close(fig)
    if point_titles:
        # Native SVG titles preserve the standalone figure and expose the
        # recorded value on hover without adding labels over the data.
        namespace="http://www.w3.org/2000/svg"
        ET.register_namespace("",namespace)
        ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
        tree=ET.parse(path)
        for group_id,titles in point_titles.items():
            group=tree.find(f".//{{{namespace}}}g[@id='{group_id}']")
            assert group is not None,group_id
            markers=group.findall(f".//{{{namespace}}}use")
            assert len(markers)==len(titles),(group_id,len(markers),len(titles))
            for marker,title in zip(markers,titles):
                ET.SubElement(marker,f"{{{namespace}}}title").text=title
        tree.write(path,encoding="utf-8",xml_declaration=True)


def parallelism(records,output,mobile):
    fig,ax=frame("Training phase time",mobile)
    point_titles={}
    for tp,color,marker in [(1,ACCENT,"o"),(2,"#576375","s"),(4,"#ae6932","^")]:
        data=sorted([r for r in records if r["tp"]==tp],key=lambda r:r["gpus"])
        label="DP, TP1" if tp==1 else f"TP{tp} + DP"
        ax.plot([r["gpus"] for r in data],[r["warm_train_seconds"] for r in data],
                color=color,marker=marker,markersize=4.5,linewidth=1.1,label=label,gid=f"series-tp{tp}")
        point_titles[f"series-tp{tp}"]=[f"{r['gpus']} B200s, TP{tp}/DP{r['gpus']//tp}: {r['warm_train_seconds']:.2f} seconds" for r in data]
    ax.set_xscale("log",base=2)
    ax.set_xticks([1,2,4,8],labels=["1","2","4","8"])
    ax.set_xlabel("B200 training GPUs",color=INK)
    ax.set_ylabel("Seconds per phase",color=INK)
    ax.set_ylim(0,135)
    ax.legend(frameon=False,fontsize=9,loc="upper right")
    save(fig,output/f"grpo-qwen35-parallelism{'-mobile' if mobile else ''}.svg",point_titles)


def learning(task,report,output,mobile,axis="responses"):
    title={"gsm8k":"GSM8K validation","dapo":"DAPO validation"}[task]
    fig,ax=frame(title,mobile)
    curve=report["validation"]
    transform={"responses":lambda r:r["responses"]/1024,"output_tokens":lambda r:r["output_tokens"]/1e6,
               "elapsed_seconds":lambda r:r["elapsed_seconds"]/3600}[axis]
    x=[transform(r) for r in curve]
    y=[100*r["accuracy"] for r in curve]
    eligible=[r for r in curve if r["responses"]>0 and r["responses"]%32768==0]
    ax.scatter([transform(r) for r in eligible],[100*r["accuracy"] for r in eligible],
               facecolors="none",edgecolors="#88929c",s=65,linewidths=1.1,zorder=2)
    ax.plot(x,y,color=ACCENT,marker="o",markersize=3.5,linewidth=1.2,zorder=3,gid="validation-points")
    labels={"responses":"Training responses (×1,024)","output_tokens":"Retained output tokens (millions)",
            "elapsed_seconds":"Elapsed job time (hours)"}
    ax.set_xlabel(labels[axis],color=INK,fontsize=10 if mobile else 11)
    ax.set_ylabel("Accuracy (%)",color=INK)
    ax.set_ylim(max(0,min(y)-1),min(100,max(y)+1))
    if axis=="responses":
        ax.set_xticks([0,32,64,96,128])
    suffix="" if axis=="responses" else "-"+axis
    titles=[f"{r['responses']:,} training responses: {100*r['accuracy']:.2f}% validation accuracy; {r['output_tokens']:,} retained output tokens; {r['elapsed_seconds']/3600:.2f} elapsed job hours" for r in curve]
    save(fig,output/f"grpo-qwen35-{task}{suffix}{'-mobile' if mobile else ''}.svg",{"validation-points":titles})


def social_card(output):
    image=Image.new("RGB",(1200,630),"#fbfbfa")
    draw=ImageDraw.Draw(image)
    fontroot=Path(matplotlib.get_data_path())/"fonts/ttf"
    def font(size,bold=False):
        return ImageFont.truetype(str(fontroot/("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")),size)
    draw.text((58,42),"GRPO batch sizes",font=font(72,True),fill=INK)
    draw.text((62,143),"Qwen3.5-4B · two math training runs",font=font(31),fill="#515963")
    for box in [(60,240,548,459),(652,240,1140,459)]:
        draw.rounded_rectangle(box,radius=20,fill="#f0f1ff",outline="#deddfb",width=2)
    draw.text((91,263),"GENERATE",font=font(26,True),fill=ACCENT)
    draw.text((91,310),"128 prompts × 8",font=font(38,True),fill=INK)
    draw.text((91,376),"1,024 responses",font=font(32),fill="#515963")
    draw.text((687,263),"TRAIN",font=font(26,True),fill=ACCENT)
    draw.text((687,310),"4 updates × 256",font=font(38,True),fill=INK)
    draw.text((687,376),"Each response once",font=font(30),fill="#515963")
    draw.line([(565,350),(634,350)],fill=ACCENT,width=5)
    draw.polygon([(634,350),(616,338),(616,362)],fill=ACCENT)
    draw.text((62,493),"Choose the learning batch. Measure GPU packing.",font=font(29),fill=INK)
    draw.text((62,580),"simondong1.github.io · Simon Dong",font=font(23),fill="#65707b")
    image.save(output/"grpo-batch-knobs-og.png")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--main-json",type=Path)
    parser.add_argument("--systems",type=Path,default=Path("artifacts/tp-dp-final.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/article-assets"))
    args=parser.parse_args()
    # Outline glyphs so standalone figures keep their measured layout even
    # when the reader's device does not have Matplotlib's font installed.
    plt.rcParams.update({"svg.fonttype":"path","font.family":"DejaVu Sans"})
    args.output.mkdir(parents=True,exist_ok=True)
    for mobile in (False,True):
        parallelism(json.loads(args.systems.read_text())["results"],args.output,mobile)
    if args.main_json:
        for task,report in json.loads(args.main_json.read_text()).items():
            assert report["completed_audit"],"Final learning figures require completed runs"
            for mobile in (False,True):
                for axis in ("responses","output_tokens","elapsed_seconds"):
                    learning(task,report,args.output,mobile,axis)
    social_card(args.output)
    print(str(args.output))


if __name__=="__main__":
    main()
