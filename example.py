from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'work/case_study_v5'))
from runtime import make_session
s=make_session('S01-a',output_dir=root/'outputs/case_study_v5')
r=s.environment.call('query_observations',{'variable':'temperature','start':'2024-08-13','end':'2024-08-27','operation':'records','filters':[{'variable':'temperature','operator':'>=','value':29},{'variable':'precipitation','operator':'<=','value':3}]})
print(json.dumps({'status':r['status'],'dates':[v['date'] for v in r.get('records',[])]},ensure_ascii=False,indent=2))
