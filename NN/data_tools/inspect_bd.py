#!/usr/bin/env python3
"""Read-only проверка контракта пользовательского Synthetic Corpus v1.

Не запускает генератор/проекты, не изменяет BD, не обучает сеть и не строит
словарь. Отчёт создаётся в NN/runs или в заданной новой папке.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]


class Corpus:
    def __init__(self,folder=None,archive=None):
        self.archive=zipfile.ZipFile(archive) if archive else None
        if self.archive:
            candidates=[n[:-len(suffix)] for n in self.archive.namelist() for suffix in ('labels/projects.jsonl','catalog/projects.jsonl') if n.endswith(suffix)]
            if len(candidates)!=1:
                raise ValueError('В архиве должен быть один корень с labels/projects.jsonl')
            self.prefix=candidates[0]
            self.structured=self.prefix+'catalog/projects.jsonl' in self.archive.namelist()
        else:
            self.folder=Path(folder)
            if not (self.folder/'labels/projects.jsonl').is_file() and not (self.folder/'catalog/projects.jsonl').is_file():
                self.folder=self.folder/'PGR_Neural_Synthetic_Corpus_v1'
            if not (self.folder/'labels/projects.jsonl').is_file() and not (self.folder/'catalog/projects.jsonl').is_file():
                raise ValueError('Не найден каталог данных. Укажите --bd.')
            self.structured=(self.folder/'catalog/projects.jsonl').is_file()

    def mapped(self,name):
        if not self.structured:
            return name
        if name.startswith('labels/'):
            base=name.split('/')[-1]
            if base in ('train.jsonl','val.jsonl','test.jsonl'):
                return 'splits/'+base
            if base=='repair_pairs.jsonl':
                return 'repairs/'+base
            return 'catalog/'+base
        if name.startswith('observations/') and name.endswith('_observations.jsonl'):
            compiler=name.split('/')[-1].split('_')[0]
            return 'observations/'+compiler+'/observations.jsonl'
        if name.startswith('training/'):
            split,compiler=Path(name).stem.split('_')
            return 'datasets/'+compiler+'/'+split+'.jsonl'
        return name

    def exists(self,name):
        name=self.mapped(name)
        if self.archive:
            return self.prefix+name in self.archive.namelist()
        return (self.folder/name).is_file()

    def read(self,name):
        name=self.mapped(name)
        if self.archive:
            return self.archive.read(self.prefix+name).decode('utf-8-sig')
        return (self.folder/name).read_text(encoding='utf-8-sig')

    def lines(self,name):
        name=self.mapped(name)
        # Потоковый разбор training: исходники всех проектов не копируются в RAM.
        if self.archive:
            stream=self.archive.open(self.prefix+name)
        else:
            stream=(self.folder/name).open('rb')
        with stream:
            for n,raw in enumerate(stream,1):
                text=raw.decode('utf-8-sig').strip()
                if text:
                    row=json.loads(text)
                    if not isinstance(row,dict):
                        raise ValueError('%s:%d не является объектом' % (name,n))
                    yield row

    def close(self):
        if self.archive:
            self.archive.close()


def inspect(c):
    taxonomy=json.loads(c.read('labels/fault_taxonomy.json'))
    if not isinstance(taxonomy,list) or not all(isinstance(r,dict) and isinstance(r.get('code'),str) for r in taxonomy):
        raise ValueError('Неверная таксономия')
    codes={r['code'] for r in taxonomy}
    if len(codes)!=len(taxonomy):
        raise ValueError('Повторный код таксономии')
    catalog={};pairs=defaultdict(list);counts=Counter();class_splits=defaultdict(set)
    for r in c.lines('labels/projects.jsonl'):
        pid=r['project_id']
        if pid in catalog:
            raise ValueError('Повторный project_id '+pid)
        if r['split'] not in ('train','val','test') or r['fault_code'] not in codes|{'CLEAN'}:
            raise ValueError('Некорректные split/fault_code '+pid)
        if not isinstance(r['expected_ok'],bool):
            raise ValueError('expected_ok должен быть bool')
        if r['expected_stage'] not in ('success','compile_error','link_error','configure_error','runtime_error'):
            raise ValueError('Неизвестная стадия '+pid)
        catalog[pid]=r;pairs[r['pair_id']].append(r);counts[r['split']]+=1
        if r['fault_code']!='CLEAN':
            class_splits[r['fault_code']].add(r['split'])
    for key,rs in pairs.items():
        if len(rs)!=2 or len({r['split'] for r in rs})!=1 or sum(r['fault_code']=='CLEAN' for r in rs)!=1:
            raise ValueError('Нарушение парного разделения '+key)
    for split in ('train','val','test'):
        expected={k for k,v in catalog.items() if v['split']==split}
        if c.exists('labels/'+split+'.jsonl'):
            actual=[r['project_id'] for r in c.lines('labels/'+split+'.jsonl')]
            if set(actual)!=expected or len(actual)!=len(set(actual)):
                raise ValueError('Несогласован labels/'+split+'.jsonl')
    training={};observations={};failed_ids={}
    forbidden={'target','fault_code','pair_id','split','expected_stage','match','patch','project_id','id'}
    for compiler in ('gcc','clang'):
        obs='observations/'+compiler+'_observations.jsonl'
        if c.exists(obs):
            seen=set();matched=0;failed_ids[compiler]=set()
            for r in c.lines(obs):
                meta=catalog.get(r['project_id'])
                if meta is None or r['project_id'] in seen:
                    raise ValueError('Неизвестный/повторный observation')
                seen.add(r['project_id'])
                if r['pair_id']!=meta['pair_id'] or r['split']!=meta['split'] or r['fault_code']!=meta['fault_code']:
                    raise ValueError('Метаданные observation не совпадают с каталогом')
                if not isinstance(r['match'],bool):
                    raise ValueError('match должен быть bool')
                matched+=int(r['match'])
                if not r['match']:
                    failed_ids[compiler].add(r['project_id'])
            observations[compiler]={'rows':len(seen),'reported_matches':matched,'missing_projects':len(set(catalog)-seen)}
        for split in ('train','val','test'):
            name='training/'+split+'_'+compiler+'.jsonl'
            if not c.exists(name):
                continue
            seen=set();empty=Counter();targets=Counter()
            for r in c.lines(name):
                pid=r['id'];meta=catalog.get(pid)
                if meta is None or pid in seen or meta['split']!=split or r['pair_id']!=meta['pair_id']:
                    raise ValueError('Несогласованность training id/split/pair')
                seen.add(pid);inp=r['input'];target=r['target']
                if not isinstance(inp,dict) or forbidden.intersection(inp):
                    raise ValueError('Прямая утечка метки в input '+pid)
                if not isinstance(inp.get('diagnostic'),str) or inp.get('compiler')!=compiler:
                    raise ValueError('Неверный input diagnostic/compiler')
                if target.get('fault_code')!=meta['fault_code'] or target.get('stage')!=meta['expected_stage'] or target.get('build_ok')!=meta['expected_ok']:
                    raise ValueError('Неверный target '+pid)
                targets[target['stage']]+=1
                if not inp['diagnostic'].strip():
                    empty[target['stage']]+=1
            expected={k for k,v in catalog.items() if v['split']==split}
            training[name]={'rows':len(seen),'missing_projects':len(expected-seen),'rows_with_failed_observation':len(seen & failed_ids.get(compiler,set())),
                            'targets':dict(targets),'empty_diagnostics_by_stage':dict(empty)}
    if not training:
        raise ValueError('Нет training/*.jsonl, сначала подготовьте корпус средствами BD')
    return {'contract':'Synthetic Corpus v1', 'storage_layout':2 if c.structured else 1,'projects':len(catalog),'taxonomy_fault_codes':len(codes),
            'pairs':len(pairs),'split_counts':dict(counts),'observations':observations,'training':training,
            'warnings':[
                'Проверена структура, не правильность меток/наблюдений или исходников; SHA256 и перекомпиляция остаются в BD.',
                'Разделение по парам не даёт независимости семейств шаблонов.',
                'В корпусе target.build_ok включает успешность выполнения; runtime_error не означает ошибку компиляции.',
                '117 целей (116+CLEAN) не сводятся автоматически к семи классам baseline NN.',
                'Идентификаторы, split, пары, target и патчи нельзя подавать в признаки модели.',
                'Чтение test здесь — аудит структуры, не настройка словаря, весов или гиперпараметров.'
            ], 'families_present_in_multiple_splits':sum(len(s)>1 for s in class_splits.values())}


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            if stream.isatty():
                stream.reconfigure(errors='backslashreplace')
            else:
                stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--bd',type=Path,default=None)
    group.add_argument('--bd-zip',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();corpus=None
    try:
        corpus=Corpus(args.bd or ROOT.parent/'BD',args.bd_zip)
        report=inspect(corpus)
        if args.output:
            output=args.output
        else:
            (ROOT/'runs').mkdir(exist_ok=True)
            output=ROOT/'runs'/('bd_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        output.mkdir(exist_ok=False)
        (output/'bd_contract_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Корпус:',report['projects'],'проектов;',report['taxonomy_fault_codes'],'типов ошибок;',report['pairs'],'пар')
        print('Контракт прочитан; отчёт:',output.resolve())
        for warning in report['warnings']:
            print('Примечание:',warning)
        return 0
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print('Ошибка контракта: '+str(exc),file=sys.stderr)
        return 2
    finally:
        if corpus:
            corpus.close()


if __name__=='__main__':
    sys.exit(main())
