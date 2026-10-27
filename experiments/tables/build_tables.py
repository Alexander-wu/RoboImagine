#!/usr/bin/env python3
"""Compile exact table bodies to a five-page standalone preview (no experiments)."""
from pathlib import Path
import shutil,subprocess,json,re,os
R=Path(__file__).resolve().parent;out=R.parent/'reproduced/tables';out.mkdir(parents=True,exist_ok=True)
latexmk=shutil.which('latexmk')
if not latexmk:
 p=Path.home()/'.local/bin/latexmk'
 if p.exists():latexmk=str(p)
if not latexmk:
 print('SKIPPED table PDF: latexmk unavailable. All LaTeX and CSV sources remain usable.')
 (out/'execution.json').write_text(json.dumps({'state':'SKIPPED','reason':'latexmk unavailable'}));raise SystemExit(0)
shutil.copy2(R/'table_style.tex',out/'table_style.tex')
pre=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=14mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[table]{xcolor}
\usepackage{booktabs,array,xparse,caption,amsmath}
\usepackage{newtxtext,newtxmath}
\definecolor{mydarkblue}{HTML}{235483}
\definecolor{codepurple}{HTML}{604193}
\definecolor{codegreen}{HTML}{397F70}
\definecolor{codegray}{HTML}{666666}
\newcommand{\roboimagine}{\textsc{RoboImagine}}
\input{table_style}
\begin{document}
'''
body=[]
for n in range(1,6):
 s=(R/f'table{n:02d}.tex').read_text()
 s=re.sub(r'\\begin\{table\*?\}\[[^]]*\]','',s);s=re.sub(r'\\end\{table\*?\}','',s)
 s=s.replace(r'\caption{',r'\captionof{table}{')
 # same half-column width for ablation/search tables; no numerical changes
 width='0.49\\textwidth' if n in [3,4] else '\\textwidth'
 body.append(r'\begin{center}\begin{minipage}{'+width+'}\n'+r'\setlength{\columnwidth}{\linewidth}'+'\n'+s+'\n'+r'\end{minipage}\end{center}\clearpage')
(out/'tables.tex').write_text(pre+'\n'.join(body)+'\n'+r'\end{document}')
with (out/'compile.log').open('w') as log:subprocess.run([latexmk,'-pdf','-interaction=nonstopmode','-halt-on-error','tables.tex'],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
(out/'execution.json').write_text(json.dumps({'state':'PASS','tables':5,'source':'current-paper LaTeX snapshots','pdf':'tables.pdf'}))
print(out/'tables.pdf')
