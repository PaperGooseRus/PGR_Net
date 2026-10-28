"""Сопоставление предварительных меток и правил v1/v2; не нейросеть."""
from dataclasses import asdict
import json
from pathlib import Path
from baseline_v1.classifier import classify as old_classify
from classifier_v2 import classify, KNOWN, VERSION
from io_utils import write_jsonl


def compare(rows):
    results = []
    for row in rows:
        if row.get('label') not in KNOWN | {'out_of_scope','build_success'}:
            raise ValueError('Неподдерживаемая метка: '+str(row.get('label')))
        expected = 'unknown' if row['label']=='out_of_scope' else row['label']
        status = row.get('status','expected_marker_found')
        eligible = row['label']!='build_success' and status=='expected_marker_found'
        item = {'id':row['id'],'label':row['label'],'expected':expected,'status':status,
                'diagnostic':row.get('diagnostic',''),'evaluated':eligible}
        if eligible:
            if not isinstance(row.get('diagnostic'),str) or not row['diagnostic'].strip():
                raise ValueError('Нет строки диагностики для '+row['id'])
            if not isinstance(row.get('stage','unknown'),str):
                raise ValueError('Некорректная stage для '+row['id'])
            new = classify(row)
            old = old_classify(row['diagnostic'],row.get('stage','unknown'))
            item.update(v1=asdict(old), v2=asdict(new), v1_matches=old.label==expected,
                        v2_matches=new.label==expected)
        results.append(item)
    tested = [r for r in results if r['evaluated']]
    summary = {'classifier_version':VERSION,'total':len(rows),'evaluated_errors':len(tested),
               'successful_controls':sum(r['label']=='build_success' and r['status']=='control_success' for r in results),
               'collection_problems':sum(r['status'] not in ('expected_marker_found','control_success') for r in results),
               'v1_matches':sum(r['v1_matches'] for r in tested),'v2_matches':sum(r['v2_matches'] for r in tested),
               'v2_known_rejected':sum(r['label']!='out_of_scope' and r['v2']['label']=='unknown' for r in tested),
               'v2_unknown_false_accepts':sum(r['label']=='out_of_scope' and r['v2']['label']!='unknown' for r in tested),
               'regressions':[r['id'] for r in tested if r['v1_matches'] and not r['v2_matches']],
               'improvements':[r['id'] for r in tested if not r['v1_matches'] and r['v2_matches']],
               'limitations':'Разработческие регрессионные данные; не независимая оценка точности и не датасет для обучения.'}
    return results, summary


def write_report(output, rows):
    results,summary=compare(rows)
    write_jsonl(Path(output)/'predictions.jsonl',results)
    (Path(output)/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['Сравнение правил v1 и v2',
           'Оценено ошибочных сборок: '+str(summary['evaluated_errors']),
           'Совпадения v1: '+str(summary['v1_matches']),
           'Совпадения v2: '+str(summary['v2_matches']),
           'Проблем сбора: '+str(summary['collection_problems']),
           'Успешных контрольных сборок: '+str(summary['successful_controls']),
           'Регрессии: '+str(summary['regressions']), '', 'Несовпадения v2:']
    for r in results:
        if r['evaluated'] and not r['v2_matches']:
            lines.append('%s: %s -> %s\n  %s' % (r['id'],r['expected'],r['v2']['label'],r['diagnostic']))
    lines += ['',summary['limitations']]
    (Path(output)/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    return summary
