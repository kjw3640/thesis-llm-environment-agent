"""Model-neutral, in-memory observation tools and scripted dialogue state.

No network client, model invocation, filesystem tools, or automatic grading.
Only the factory reads evaluator assets; each exposed session owns one packet.
"""
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
import json
import math
import operator
from pathlib import Path

OPS = {'>':operator.gt,'>=':operator.ge,'<':operator.lt,'<=':operator.le,'=':operator.eq,'==':operator.eq,'!=':operator.ne}

def error(code, message, *, kind='invalid_request', retryable=False, **details):
    return {'status':'error','code':code,'kind':kind,'message':message,'retryable':retryable,**details}

def decimal(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric value')
    return Decimal(str(value))

def dates(start,end):
    a,b=date.fromisoformat(start),date.fromisoformat(end)
    if a>b or (b-a).days>3660:
        raise ValueError('Invalid or excessive date range')
    return [(a+timedelta(days=i)).isoformat() for i in range((b-a).days+1)]

class ObservationSession:
    def __init__(self, packet, state='normal', max_calls=16):
        if state not in ('normal','transient_once','requires_selection','permanent_failure','no_file_export'):
            raise ValueError('Unknown execution state')
        self._packet=deepcopy(packet)
        self.state=state
        self.selected=state!='requires_selection'
        self.transient_used=False
        self.max_calls=max_calls
        self.events=[]

    def public_metadata(self):
        return {'dataset_id':self._packet['packet_id'],'metadata':deepcopy(self._packet['metadata'])}

    def call(self,name,args):
        if len(self.events)>=self.max_calls:
            return error('CALL_BUDGET','Tool-call budget exhausted',kind='execution_limit')
        if not isinstance(args,dict):
            result=error('INVALID_ARGUMENTS','Arguments must be an object')
        else:
            try:
                if name=='select_dataset':
                    if set(args)!={'dataset_id'} or args['dataset_id']!=self._packet['packet_id']:
                        result=error('DATASET_NOT_AVAILABLE','Only the current dataset is accessible',kind='data_limitation')
                    else:
                        self.selected=True
                        result={'status':'ok','dataset_id':self._packet['packet_id'],'selected':True}
                elif name=='subtract':
                    if set(args)!={'left','right'}:
                        raise ValueError('subtract requires only left and right')
                    result={'status':'ok','value':float(decimal(args['left'])-decimal(args['right']))}
                elif name=='query_observations':
                    result=self._query(args)
                else:
                    result=error('FUNCTION_NOT_AVAILABLE','This function is not available',kind='capability_limitation')
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                result=error('INVALID_ARGUMENTS',str(exc))
        self.events.append({'sequence':len(self.events)+1,'name':name,'arguments':deepcopy(args),'result':deepcopy(result)})
        return result

    def _query(self,args):
        allowed={'variable','start','end','source_id','station','operation','filters','sort_by','descending','limit','missing_policy'}
        if set(args)-allowed or not {'variable','start','end','operation'}<=set(args):
            raise ValueError('Missing required or unknown query arguments')
        requested=dates(args['start'],args['end'])
        md=self._packet['metadata']
        var=args['variable']
        if var not in md['variables']:
            return error('VARIABLE_NOT_AVAILABLE','Variable is absent from this dataset',kind='data_limitation',available_variables=list(md['variables']))
        if args.get('station',md['station'])!=md['station']:
            return error('STATION_NOT_AVAILABLE','Requested station is not in this dataset',kind='data_limitation',available_stations=md['available_stations'])
        source_id=args.get('source_id')
        if source_id is None:
            if len(self._packet['sources'])!=1:
                return error('SOURCE_REQUIRED','Choose one of the listed source identifiers',available_sources=md['source_ids'])
            source_id=md['source_ids'][0]
        src=next((s for s in self._packet['sources'] if s['source_id']==source_id),None)
        if src is None:
            return error('SOURCE_NOT_AVAILABLE','Source identifier is unavailable',kind='data_limitation')
        op=args['operation']
        missing_policy=args.get('missing_policy','strict')
        if op not in ('records','sum','mean') or missing_policy not in ('strict','available_only'):
            raise ValueError('Invalid operation or missing policy')
        filters=args.get('filters',[])
        if not isinstance(filters,list):
            raise ValueError('filters must be an array')
        for f in filters:
            if not isinstance(f,dict) or set(f)!={'variable','operator','value'} or f['variable'] not in md['variables'] or f['operator'] not in OPS:
                raise ValueError('Invalid filter specification')
            decimal(f['value'])
        sort=args.get('sort_by')
        if sort is not None and sort not in [*md['variables'],'date']:
            raise ValueError('Invalid sort field')
        descending=args.get('descending',False)
        limit=args.get('limit')
        if not isinstance(descending,bool) or (limit is not None and (isinstance(limit,bool) or not isinstance(limit,int) or limit<=0)):
            raise ValueError('Invalid descending or limit')
        # Fault injection follows argument validation; malformed calls do not consume it.
        if not self.selected:
            return error('DATASET_NOT_SELECTED','Call select_dataset with the current dataset_id, then query again',kind='precondition',dataset_id=self._packet['packet_id'])
        if self.state=='permanent_failure':
            return error('SERVICE_UNAVAILABLE','Observation query service is unavailable for this session',kind='execution_failure',retryable=False)
        if self.state=='transient_once' and not self.transient_used:
            self.transient_used=True
            return error('SERVICE_TEMPORARY','Temporary query-service failure; retry is allowed',kind='execution_failure',retryable=True)
        raw=[deepcopy(r) for r in src['rows'] if args['start']<=r['date']<=args['end']]
        present={r['date'] for r in raw}
        absent=sorted(set(requested)-present)
        info={'source_id':source_id,'dataset_id':self._packet['packet_id'],'variable':var,
              'unit':md['variables'][var]['stored_unit'],'canonical_multiplier':md['variables'][var]['canonical_multiplier'],
              'requested_count':len(requested),'input_row_count':len(raw),'missing_calendar_dates':absent}
        if not raw:
            return error('DATES_NOT_AVAILABLE','No requested dates are available',kind='data_limitation',**info)
        # Filtering/sorting cannot safely ignore missing or flagged selector inputs.
        selectors={f['variable'] for f in filters}|({sort} if sort and sort!='date' else set())
        selector_bad=sorted({r['date'] for r in raw for v in selectors if not self._eligible(r,v)})
        if selector_bad:
            return error('SELECTION_INDETERMINATE','Missing or QC-invalid filter/sort values',kind='data_limitation',indeterminate_dates=selector_bad,**info)
        chosen=raw
        for f in filters:
            chosen=[r for r in chosen if OPS[f['operator']](decimal(r[f['variable']]),decimal(f['value']))]
        # Stable date order breaks numeric ties before limiting.
        chosen.sort(key=lambda r:r['date'])
        if sort:
            chosen.sort(key=lambda r:r[sort],reverse=descending)
        if op=='records':
            if limit is not None:
                chosen=chosen[:limit]
            return {'status':'ok','operation':op,'records':chosen,'selected_row_count':len(chosen),
                    'selected_unique_count':len({r['date'] for r in chosen}),
                    'observed_unique_count':len({r['date'] for r in raw if self._eligible(r,var)}),**info}
        # Aggregate one observation per key, not physical file row. Reject conflicts.
        buckets={}
        relevant={var}|selectors
        for r in chosen:
            key=r['date']
            signature=tuple((v,r[v],r['quality_flags'].get(v,''),r['measurement_flags'].get(v,'')) for v in sorted(relevant))
            if key in buckets and buckets[key][0]!=signature:
                return error('DUPLICATE_VALUE_CONFLICT','Conflicting duplicate observations need explicit resolution',kind='data_limitation',date=key,**info)
            buckets.setdefault(key,(signature,r))
        duplicate_count=len(chosen)-len(buckets)
        unique=[v[1] for v in buckets.values()]
        unique.sort(key=lambda r:r['date'])
        if sort:
            unique.sort(key=lambda r:r[sort],reverse=descending)
        if limit is not None:
            unique=unique[:limit]
        excluded=sorted(r['date'] for r in unique if not self._eligible(r,var))
        good=[r for r in unique if self._eligible(r,var)]
        incomplete=bool(absent or excluded)
        stats={'selected_unique_count':len(unique),'observed_unique_count':len(good),'duplicate_rows_removed':duplicate_count,
               'excluded_dates':excluded,'selected_dates':[r['date'] for r in unique],
               'eligible_dates':[r['date'] for r in good],'partial':incomplete,**info}
        if incomplete and missing_policy=='strict':
            return error('INCOMPLETE_DATA','Full aggregate is not defined; request available_only for an explicit observed subset',kind='data_limitation',**stats)
        total=sum((decimal(r[var]) for r in good),Decimal(0))
        value=total if op=='sum' else total/len(good) if good else None
        return {'status':'ok','operation':op,'value':float(value) if value is not None else None,
                'value_defined':value is not None,**stats}

    @staticmethod
    def _eligible(r,var):
        return r[var] is not None and not r['quality_flags'].get(var,'').strip() and r['measurement_flags'].get(var,'')!='P'

    def recovery_observations(self):
        errors=[e for e in self.events if e['result'].get('kind')=='execution_failure' or e['result'].get('code')=='DATASET_NOT_SELECTED']
        recovered=any(e['name']=='query_observations' and e['result']['status']=='ok' and e['sequence']>bad['sequence'] for bad in errors for e in self.events)
        return {'error_exposed':bool(errors),'query_recovered':recovered if errors else None,
                'prerequisite_fulfilled':self.selected if self.state=='requires_selection' else None}

class ScriptedConversation:
    """Accept supplied assistant messages; never generate one or infer correctness."""
    def __init__(self, packet, task, system_prompt):
        self.environment=ObservationSession(packet,task['tool_environment'])
        self._followup=task['followup']
        self._followup_sent=False
        self._awaiting_tools=False
        self.messages=[{'role':'system','content':system_prompt},
                       {'role':'user','content':json.dumps(self.environment.public_metadata(),ensure_ascii=False)+'\n질문: '+task['question']}]

    def assistant(self,message):
        if self._awaiting_tools:
            raise ValueError('Complete tool results before accepting another assistant message')
        if message.get('role')!='assistant':
            raise ValueError('Expected assistant message')
        self.messages.append(deepcopy(message))
        calls=message.get('tool_calls') or []
        self._awaiting_tools=bool(calls)
        for call in calls:
            fn=call['function']
            args=fn.get('arguments',{})
            if isinstance(args,str):
                try:
                    args=json.loads(args)
                except ValueError:
                    args=None
            result=self.environment.call(fn['name'],args)
            self.messages.append({'role':'tool','tool_name':fn['name'],'tool_call_id':call.get('id'),
                                  'content':json.dumps(result,ensure_ascii=False)})
        self._awaiting_tools=False

    def next_user_turn(self):
        if not self._followup or self._followup_sent or not self.messages or self.messages[-1]['role']!='assistant' or self.messages[-1].get('tool_calls'):
            raise ValueError('A final assistant response must precede the scripted follow-up')
        self.messages.append({'role':'user','content':self._followup})
        self._followup_sent=True

def make_session(case_id, output_dir=None):
    """Evaluator-only factory; do not expose this function as a model tool."""
    out=Path(output_dir) if output_dir else Path(__file__).resolve().parents[2]/'outputs/case_study_v5'
    tasks=json.loads((out/'tasks.json').read_text())
    task=next(t for t in tasks if t['id']==case_id)
    packet=json.loads((out/'packets.json').read_text())[task['packet_id']]
    design=json.loads((out/'design.json').read_text())
    # The exposed instance receives the current packet, not the task reference or task list.
    return ScriptedConversation(packet,{k:task[k] for k in ('question','tool_environment','followup')},design['system_prompt'])
