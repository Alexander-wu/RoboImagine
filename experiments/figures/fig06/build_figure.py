#!/usr/bin/env python3
"""Compose saved frames and consistently cropped detail panels; no retouching."""
from pathlib import Path
import os,json,hashlib
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patheffects
from matplotlib.patches import Rectangle,ConnectionPatch
from PIL import Image
for name in ['Regular','Bold','Italic','BoldItalic']:
 font_manager.fontManager.addfont(str(HERE/'fonts'/('EBGaramond-'+name+'.otf')))
plt.rcParams.update({'font.family':'EB Garamond','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#283040'})
selection=json.loads((HERE/'selection.json').read_text())
rois={'RT-1':[150,80,278,208],'BridgeV2':[130, 100, 250, 220],'RoboCasa':[180,115,320,255]}
# Four full frames plus a square detail per dataset; full row names at right.
W=7.16
LEFT,RIGHT,PANEL_GAP=.012,.34,.10
PANEL_W=(W-LEFT-RIGHT-PANEL_GAP)/2
FRAME_GAP,DETAIL_GAP=.016,.045
IMAGE_W=(PANEL_W-3*FRAME_GAP-DETAIL_GAP)/4.8
IMAGE_H=IMAGE_W*.8
ROW_GAP,BOTTOM,HEADER=.035,.032,.29
H=BOTTOM+5*IMAGE_H+4*ROW_GAP+HEADER
fig=plt.figure(figsize=(W,H),facecolor='white')
ink,purple,gold='#293246','#6E528B','#B98B36'
halo=[patheffects.Stroke(linewidth=1.8,foreground='white'),patheffects.Normal()]
audit=[]
def image_axes(x,y,width,height,image,border='#D8D9DE',thickness=.35):
 ax=fig.add_axes([x/W,y/H,width/W,height/H])
 ax.imshow(image,interpolation='none');ax.set_axis_off()
 ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,linewidth=thickness,edgecolor=border,clip_on=False,zorder=5))
 return ax
for column,item in enumerate(selection):
 x=LEFT+column*(PANEL_W+PANEL_GAP);dataset=item['dataset']
 fig.text((x+PANEL_W/2)/W,(H-.090)/H,f'({chr(97+column)}) {dataset} · '+('Bag' if dataset=='RT-1' else 'Spoon'),ha='center',va='center',fontsize=9,weight='bold',color=ink)
 for k,frame in enumerate([8,16,24,32]):
  fig.text((x+k*(IMAGE_W+FRAME_GAP)+IMAGE_W/2)/W,(H-.213)/H,f't = {frame}',ha='center',va='center',fontsize=7.4,color=ink)
 zoom_x=x+4*IMAGE_W+3*FRAME_GAP+DETAIL_GAP
 fig.text((zoom_x+IMAGE_H/2)/W,(H-.213)/H,'Detail',ha='center',va='center',fontsize=7.4,color=ink)
 box=rois[dataset];x0,y0,x1,y1=box
 assert x1-x0==y1-y0 and 0<=x0<x1<=320 and 0<=y0<y1<=256
 for row,key in enumerate(['GT','Cosmos-Predict2.5','Ctrl-World','WEAVER','R2']):
  y=BOTTOM+(4-row)*(IMAGE_H+ROW_GAP)
  if column==1:
   label={'GT':'Ground truth','Cosmos-Predict2.5':'Cosmos-\nPredict2.5-2B','Ctrl-World':'Ctrl-World','WEAVER':'WEAVER','R2':'RoboImagine'}[key]
   fig.text((W-RIGHT/2)/W,(y+IMAGE_H/2)/H,label,ha='center',va='center',rotation=90,fontsize=7.0,weight='bold',color=purple if key=='R2' else ink)
  for k,asset_index in enumerate([0,1,2,3]):
   asset=item['assets'][key][asset_index]
   path=HERE/asset['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256']
   im=Image.open(path).convert('RGB');assert im.size==(320,256)
   ax=image_axes(x+k*(IMAGE_W+FRAME_GAP),y,IMAGE_W,IMAGE_H,im)
   if k==3:
    ax.add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor=gold,linewidth=.65,zorder=7,path_effects=halo))
    detail=im.crop(tuple(box));zoom=image_axes(zoom_x,y,IMAGE_H,IMAGE_H,detail,gold,.65)
    for source,dest in [((x1-.5,y0-.5),(0,1)),((x1-.5,y1-.5),(0,0))]:
     fig.add_artist(ConnectionPatch(source,dest,'data','axes fraction',axesA=ax,axesB=zoom,color=gold,linewidth=.65,clip_on=False,zorder=4,path_effects=halo))
    audit.append({'dataset':dataset,'row':key,'source':asset['file'],'frame':32,'roi_xyxy':box,'crop_pixel_sha256':hashlib.sha256(detail.tobytes()).hexdigest()})
fig.canvas.draw()
for text in fig.texts:
 bounds=text.get_window_extent(fig.canvas.get_renderer())
 assert bounds.x0>=0 and bounds.y0>=0 and bounds.x1<=fig.bbox.width and bounds.y1<=fig.bbox.height,(text.get_text(),bounds)
for ext in ['pdf','svg','png']:
 fig.savefig(HERE/f'baseline_comparison.{ext}',dpi=450,facecolor='white')
(HERE/'render_metadata.json').write_text(json.dumps({'figure_inches':[W,H],'font':'EB Garamond','frames':[8,16,24,32],'full_frames':'Uncropped, original aspect ratio','detail_panels':'Frame 32; identical square ROI across GT and all four models within each dataset','frame_color_editing':'None','row_labels':['Ground truth','Cosmos-Predict2.5-2B','Ctrl-World','WEAVER','RoboImagine'],'row_label_position':'right, rotated 90 degrees','detail_audit':audit,'selection_sha256':hashlib.sha256((HERE/'selection.json').read_bytes()).hexdigest()},indent=2)+'\n')
print(f'Rendered {W:.2f} x {H:.3f} in; matched ROI panels: {len(audit)}')
