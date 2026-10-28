#!/usr/bin/env python3
"""Reproducible PGR synthetic C/C++ diagnostic project corpus. Python 3.9+."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Fault:
    code: str
    lang: str
    stage: str
    before_bad: str = ''
    before_good: str = ''
    bad: str = ''
    good: str = ''
    flags: tuple = ()
    mode: str = 'plain'

F=[]
def add(code, lang='c', stage='compile_error', before_bad='', before_good=None, bad='', good='', flags=(), mode='plain'):
    F.append(Fault(code,lang,stage,before_bad,before_bad if before_good is None else before_good,bad,good,tuple(flags),mode))

# Compile-time C faults: the good side always makes the smallest practical repair.
add('C_UNDECLARED',bad='int x = ghost_{v} + seed_{v}; (void)x;',good='int x = seed_{v} + 1; (void)x;')
add('C_SEMICOLON',bad='int x = seed_{v} + 1\n (void)x;',good='int x = seed_{v} + 1;\n (void)x;')
add('C_UNKNOWN_TYPE',bad='quantity_{v} x = 1; (void)x;',good='int x = 1; (void)x;')
add('C_BAD_HEADER',before_bad='#include "fixture_absent_{v}.h"',before_good='#include <stddef.h>',bad='(void)seed_{v};',good='(void)seed_{v};')
add('C_UNTERMINATED_STRING',bad='const char *s = "abc; (void)s;',good='const char *s = "abc"; (void)s;')
add('C_INVALID_TOKEN',bad='int x = seed_{v} @ 3; (void)x;',good='int x = seed_{v} + 3; (void)x;')
add('C_MACRO_ARITY',before_bad='#define MIX_{v}(a,b) ((a)+(b))',bad='int x = MIX_{v}(seed_{v}); (void)x;',good='int x = MIX_{v}(seed_{v}, 2); (void)x;')
add('C_PREPROC_UNCLOSED',bad='#if 1\nint x=seed_{v}; (void)x;',good='#if 1\nint x=seed_{v}; (void)x;\n#endif')
add('C_DUPLICATE_CASE',bad='switch(seed_{v}) { case 1: break; case 1: break; default: break; }',good='switch(seed_{v}) { case 1: break; case 2: break; default: break; }')
add('C_INVALID_BREAK',bad='break;',good='do { break; } while (0);')
add('C_INVALID_CONTINUE',bad='continue;',good='do { continue; } while (0);')
add('C_RETURN_IN_VOID',before_bad='void helper_{v}(void) { return 7; }',before_good='void helper_{v}(void) { return; }',bad='helper_{v}();',good='helper_{v}();')
add('C_TOO_FEW_ARGUMENTS',before_bad='int helper_{v}(int a,int b) { return a+b; }',bad='int x=helper_{v}(1); (void)x;',good='int x=helper_{v}(1,2); (void)x;')
add('C_TOO_MANY_ARGUMENTS',before_bad='int helper_{v}(int a) { return a+1; }',bad='int x=helper_{v}(1,2); (void)x;',good='int x=helper_{v}(1); (void)x;')
add('C_CONFLICTING_TYPES',before_bad='int helper_{v}(int a);\ndouble helper_{v}(int a) { return a+1; }',before_good='int helper_{v}(int a);\nint helper_{v}(int a) { return a+1; }',bad='(void)helper_{v}(seed_{v});',good='(void)helper_{v}(seed_{v});')
add('C_MISSING_FIELD',before_bad='struct Item_{v} { int real; };',bad='struct Item_{v} a = {1}; int x=a.missing; (void)x;',good='struct Item_{v} a = {1}; int x=a.real; (void)x;')
add('C_DOT_ARROW_MISMATCH',before_bad='struct Item_{v} { int real; };',bad='struct Item_{v} a = {1}; int x=a->real; (void)x;',good='struct Item_{v} a = {1}; int x=a.real; (void)x;')
add('C_NEGATIVE_ARRAY',bad='int a[-3]; (void)a;',good='int a[3]; (void)a;')
add('C_STATIC_ASSERT',before_bad='_Static_assert(0, "invalid check");',before_good='_Static_assert(1, "valid check");',bad='(void)seed_{v};',good='(void)seed_{v};')
add('C_CONST_ASSIGN',bad='const int x=seed_{v}; x=3;',good='int x=seed_{v}; x=3; (void)x;')
add('C_REDEFINE_LOCAL',bad='int x=1; int x=2; (void)x;',good='int x=1; x=2; (void)x;')
add('C_NON_LVALUE',bad='(seed_{v}+1)=3;',good='seed_{v}=3;')
add('C_BAD_INT_POINTER',bad='int *p=seed_{v}; (void)p;',good='int *p=&seed_{v}; (void)p;')
add('C_INCOMPATIBLE_POINTERS',bad='float f=1; int *p=&f; (void)p;',good='float f=1; float *p=&f; (void)p;')
add('C_MISSING_PAREN',bad='if (seed_{v} > 0 { seed_{v}++; }',good='if (seed_{v} > 0) { seed_{v}++; }')
add('C_BAD_DESIGNATOR',before_bad='struct Point_{v} { int x; };',bad='struct Point_{v} a={.y=2}; (void)a;',good='struct Point_{v} a={.x=2}; (void)a;')
add('C_INVALID_RETURN_VALUE',before_bad='int helper_{v}(void) { return "oops"; }',before_good='int helper_{v}(void) { return 1; }',bad='(void)helper_{v}();',good='(void)helper_{v}();')
add('C_PRINTF_FORMAT',before_bad='#include <stdio.h>',bad='(void)printf("%s",seed_{v});',good='(void)printf("%d",seed_{v});')
add('C_UNKNOWN_ANGLE_INCLUDE',before_bad='#include <pgr_absent_{v}.h>',before_good='#include <stddef.h>',bad='(void)seed_{v};',good='(void)seed_{v};')
add('C_WRONG_TYPEDEF',before_bad='typedef int metric_{v};',bad='metri_{v} x=1; (void)x;',good='metric_{v} x=1; (void)x;')
add('C_INVALID_FILE_REGISTER',before_bad='register int value_{v};',before_good='static int value_{v};',bad='(void)value_{v};',good='(void)value_{v};')
add('C_INCOMPLETE_STRUCT',before_bad='struct Item_{v};',before_good='struct Item_{v} { int value; };',bad='struct Item_{v} x; (void)x;',good='struct Item_{v} x={1}; (void)x;')
add('C_NONCONST_GLOBAL_INIT',before_bad='int helper_{v}(void){return 3;}\nint value_{v}=helper_{v}();',before_good='int helper_{v}(void){return 3;}\nint value_{v}=3;',bad='(void)value_{v}; (void)helper_{v}();',good='(void)value_{v}; (void)helper_{v}();')
add('C_EXCESS_SCALAR_INIT',bad='int x={1,2}; (void)x;',good='int x={1}; (void)x;')
add('C_GOTO_UNDEFINED',bad='goto missing_{v}; done_{v}: (void)seed_{v};',good='goto done_{v}; done_{v}: (void)seed_{v};')
add('C_NONINT_CASE',bad='switch(seed_{v}){case "text":break;default:break;}',good='switch(seed_{v}){case 2:break;default:break;}')
add('C_FUNCTION_PTR_TYPE',before_bad='int helper_{v}(int a){return a;}',bad='double (*p)(double)=helper_{v}; (void)p;',good='int (*p)(int)=helper_{v}; (void)p;')
add('C_REDEFINE_STRUCT',before_bad='struct Item_{v}{int a;}; struct Item_{v}{int b;};',before_good='struct Item_{v}{int a;};',bad='struct Item_{v} x={1}; (void)x;',good='struct Item_{v} x={1}; (void)x;')
add('C_CPP_NULLPTR',bad='void *p=nullptr; (void)p;',good='void *p=NULL; (void)p;')
add('C_EMPTY_CHAR',bad="int x=''; (void)x;",good="int x='a'; (void)x;")
add('C_INVALID_ARRAY_INDEX_TYPE',bad='int a[3]={1,2,3}; int x=a["2"]; (void)x;',good='int a[3]={1,2,3}; int x=a[2]; (void)x;')
add('C_UNKNOWN_PRAGMA_WARNING',bad='#error PGR_FORCED_DIAGNOSTIC_{v}',good='(void)seed_{v};')
add('C_TOO_MANY_STRUCT_FIELDS',before_bad='struct Pair_{v}{int x; int y;};',bad='struct Pair_{v} a={1,2,3}; (void)a;',good='struct Pair_{v} a={1,2}; (void)a;')

# C++-specific semantic, template, overload and type-system faults.
add('CPP_UNDECLARED',lang='cpp',bad='int a=missing_{v}; (void)a;',good='int a=seed_{v}; (void)a;')
add('CPP_MISSING_VECTOR_INCLUDE',lang='cpp',before_bad='',before_good='#include <vector>',bad='std::vector<int> a{1,2}; (void)a;',good='std::vector<int> a{1,2}; (void)a;')
add('CPP_DELETED_CONSTRUCTOR',lang='cpp',before_bad='struct Thing_{v}{ Thing_{v}()=delete; };',before_good='struct Thing_{v}{ Thing_{v}()=default; };',bad='Thing_{v} a; (void)a;',good='Thing_{v} a; (void)a;')
add('CPP_PRIVATE_MEMBER',lang='cpp',before_bad='class Thing_{v}{ int value=1; };',before_good='class Thing_{v}{ public: int value=1; };',bad='Thing_{v} a; (void)a.value;',good='Thing_{v} a; (void)a.value;')
add('CPP_ABSTRACT_INSTANCE',lang='cpp',before_bad='struct Shape_{v}{ virtual int size()=0; };',before_good='struct Shape_{v}{ virtual int size(){return 1;} };',bad='Shape_{v} a; (void)a;',good='Shape_{v} a; (void)a;')
add('CPP_OVERRIDE_MISMATCH',lang='cpp',before_bad='struct Base_{v}{ virtual void f(int){} }; struct Child_{v}:Base_{v}{ void f(double) override{} };',before_good='struct Base_{v}{ virtual void f(int){} }; struct Child_{v}:Base_{v}{ void f(int) override{} };',bad='Child_{v} a; (void)a;',good='Child_{v} a; (void)a;')
add('CPP_FINAL_INHERITANCE',lang='cpp',before_bad='struct Base_{v} final {}; struct Child_{v}:Base_{v}{};',before_good='struct Base_{v} {}; struct Child_{v}:Base_{v}{};',bad='Child_{v} a; (void)a;',good='Child_{v} a; (void)a;')
add('CPP_NO_CONSTRUCTOR',lang='cpp',before_bad='struct Thing_{v}{ Thing_{v}(int){} };',bad='Thing_{v} a;',good='Thing_{v} a(3); (void)a;')
add('CPP_AMBIGUOUS_OVERLOAD',lang='cpp',before_bad='int helper_{v}(long){return 1;} int helper_{v}(double){return 2;}',bad='int a=helper_{v}(3); (void)a;',good='int a=helper_{v}(3L); (void)a;')
add('CPP_BAD_CONVERSION',lang='cpp',bad='int a="text"; (void)a;',good='const char *a="text"; (void)a;')
add('CPP_NARROWING_LIST',lang='cpp',bad='int a{3.75}; (void)a;',good='double a{3.75}; (void)a;')
add('CPP_UNKNOWN_MEMBER',lang='cpp',before_bad='struct Thing_{v}{int real=1;};',bad='Thing_{v} a; int x=a.fake; (void)x;',good='Thing_{v} a; int x=a.real; (void)x;')
add('CPP_DEPENDENT_TYPENAME',lang='cpp',before_bad='template <class T> struct Box_{v}{ using type=int; }; template <class T> int helper_{v}(){ T::type n=1; return n; }',before_good='template <class T> struct Box_{v}{ using type=int; }; template <class T> int helper_{v}(){ typename T::type n=1; return n; }',bad='return helper_{v}<Box_{v}<int>>() == 1 ? 0 : 1;',good='return helper_{v}<Box_{v}<int>>() == 1 ? 0 : 1;')
add('CPP_WRONG_TEMPLATE_ARITY',lang='cpp',before_bad='template<class T,class U> struct Pair_{v}{};',bad='Pair_{v}<int> a; (void)a;',good='Pair_{v}<int,int> a; (void)a;')
add('CPP_STATIC_ASSERT',lang='cpp',before_bad='static_assert(false, "fixture failed");',before_good='static_assert(true, "fixture passed");',bad='(void)seed_{v};',good='(void)seed_{v};')
add('CPP_NONCOPYABLE_COPY',lang='cpp',before_bad='struct Thing_{v}{ Thing_{v}()=default; Thing_{v}(const Thing_{v}&)=delete; };',bad='Thing_{v} a; Thing_{v} b=a; (void)b;',good='Thing_{v} a; Thing_{v} &b=a; (void)b;')
add('CPP_MUTATE_CONST_METHOD',lang='cpp',before_bad='struct Thing_{v}{int data=1; void set() const {data=3;} };',before_good='struct Thing_{v}{mutable int data=1; void set() const {data=3;} };',bad='Thing_{v} a; a.set();',good='Thing_{v} a; a.set();')
add('CPP_BAD_NAMESPACE',lang='cpp',before_bad='namespace space_{v}{int real(){return 1;}}',bad='int a=space_{v}::fake(); (void)a;',good='int a=space_{v}::real(); (void)a;')
add('CPP_NO_MATCHING_OVERLOAD',lang='cpp',before_bad='int helper_{v}(int a){return a;}',bad='int x=helper_{v}("oops"); (void)x;',good='int x=helper_{v}(3); (void)x;')
add('CPP_CATCH_WITHOUT_TRY',lang='cpp',bad='catch (...) { seed_{v}++; }',good='try { seed_{v}++; } catch (...) {}')
add('CPP_MISSING_CLASS_SEMICOLON',lang='cpp',before_bad='struct Thing_{v}{int value;} ',before_good='struct Thing_{v}{int value;};',bad='Thing_{v} a{3}; (void)a;',good='Thing_{v} a{3}; (void)a;')
add('CPP_BAD_BASE_CONSTRUCTOR',lang='cpp',before_bad='struct Parent_{v}{explicit Parent_{v}(int){} }; struct Child_{v}:Parent_{v}{Child_{v}():Parent_{v}(){} };',before_good='struct Parent_{v}{explicit Parent_{v}(int){} }; struct Child_{v}:Parent_{v}{Child_{v}():Parent_{v}(1){} };',bad='Child_{v} a; (void)a;',good='Child_{v} a; (void)a;')
add('CPP_OPERATOR_UNDEFINED',lang='cpp',before_bad='struct Thing_{v}{int n=1;};',before_good='struct Thing_{v}{int n=1;}; Thing_{v} operator+(Thing_{v} a,Thing_{v} b){return {a.n+b.n};}',bad='Thing_{v} a,b; auto c=a+b; (void)c;',good='Thing_{v} a,b; auto c=a+b; (void)c;')
add('CPP_REFERENCE_NO_INIT',lang='cpp',bad='int &r; (void)r;',good='int &r=seed_{v}; (void)r;')
add('CPP_RVALUE_TO_LREF',lang='cpp',bad='int &r=3; (void)r;',good='const int &r=3; (void)r;')
add('CPP_VOID_TO_INT',lang='cpp',before_bad='void helper_{v}() {}',bad='int a=helper_{v}(); (void)a;',good='helper_{v}();')
add('CPP_DUPLICATE_CASE',lang='cpp',bad='switch(seed_{v}){case 1:break;case 1:break;default:break;}',good='switch(seed_{v}){case 1:break;case 2:break;default:break;}')
add('CPP_CONST_ASSIGN',lang='cpp',bad='const int a=seed_{v}; a=2;',good='int a=seed_{v}; a=2; (void)a;')
add('CPP_MISSING_SEMICOLON',lang='cpp',bad='int a=1\n(void)a;',good='int a=1;\n(void)a;')
add('CPP_BAD_TEMPLATE_MEMBER',lang='cpp',before_bad='template <class T> struct Box_{v}{T item;};',bad='Box_{v}<int> b{1}; (void)b.other;',good='Box_{v}<int> b{1}; (void)b.item;')
add('CPP_INVALID_DYNAMIC_CAST',lang='cpp',before_bad='struct Base_{v}{}; struct Child_{v}:Base_{v}{};',bad='Base_{v} b; auto p=dynamic_cast<Child_{v}*>(&b); (void)p;',good='Base_{v} b; auto p=static_cast<Base_{v}*>(&b); (void)p;')
add('CPP_INVALID_SPECIFIER',lang='cpp',before_bad='virtual int helper_{v}(){return 1;}',before_good='int helper_{v}(){return 1;}',bad='(void)helper_{v}();',good='(void)helper_{v}();')
add('CPP_BAD_INITIALIZER_LIST',lang='cpp',bad='int a[]={1,2,3}; int x=a[1,2]; (void)x;',good='int a[]={1,2,3}; int x=a[2]; (void)x;') # warning may be tolerated? replace later
add('CPP_MISSING_TYPENAME_STD',lang='cpp',bad='std::string a="x"; (void)a;',before_bad='',before_good='#include <string>',good='std::string a="x"; (void)a;')
add('CPP_NO_MATCHING_FUNCTION',lang='cpp',before_bad='int f_{v}(int,int){return 1;}',bad='auto x=f_{v}(1,2,3); (void)x;',good='auto x=f_{v}(1,2); (void)x;')
# Link-stage faults. All sources are minimal and require neither Qt nor downloads.
for code, lang, mode in [
    ('LINK_UNDEFINED_C','c','undefined'), ('LINK_UNDEFINED_CPP','cpp','undefined'),
    ('LINK_NO_MAIN_C','c','no_main'),('LINK_NO_MAIN_CPP','cpp','no_main'),
    ('LINK_DUPLICATE_MAIN_C','c','two_mains'),('LINK_DUPLICATE_MAIN_CPP','cpp','two_mains'),
    ('LINK_MULTIPLE_GLOBAL_C','c','two_globals'),('LINK_MULTIPLE_GLOBAL_CPP','cpp','two_globals'),
    ('LINK_MISSING_TRANSLATION_UNIT_C','c','missing_tu'),('LINK_MISSING_TRANSLATION_UNIT_CPP','cpp','missing_tu'),
    ('LINK_UNDEFINED_STATIC_CPP','cpp','static_member'),('LINK_UNDEFINED_VIRTUAL_CPP','cpp','virtual'),
    ('LINK_C_ABI_CPP','cpp','abi'),('LINK_MISSING_LIBRARY_C','c','library'),
    ('LINK_MISSING_LIBRARY_CPP','cpp','library'),('LINK_UNINSTANTIATED_TEMPLATE_CPP','cpp','template')]:
    add(code,lang=lang,stage='link_error',mode=mode)

# Runtime failures are explicit and deterministic; intentionally no hangs, UB, network or filesystem writes.
add('RUN_NONZERO_C',stage='runtime_error',bad='return 13;',good='return 0;')
add('RUN_NONZERO_CPP',lang='cpp',stage='runtime_error',bad='return 13;',good='return 0;')
add('RUN_ASSERT_C',stage='runtime_error',before_bad='#include <assert.h>',bad='assert(seed_{v}==0);',good='assert(seed_{v}>0);')
add('RUN_ASSERT_CPP',lang='cpp',stage='runtime_error',before_bad='#include <cassert>',bad='assert(seed_{v}==0);',good='assert(seed_{v}>0);')
add('RUN_UNCAUGHT_THROW_CPP',lang='cpp',stage='runtime_error',before_bad='#include <stdexcept>',bad='throw std::runtime_error("pgr fixture");',good='try {throw std::runtime_error("pgr fixture");} catch(const std::runtime_error&){}')
add('RUN_INVALID_ARGUMENT_CPP',lang='cpp',stage='runtime_error',before_bad='#include <stdexcept>',bad='throw std::invalid_argument("bad input");',good='try {throw std::invalid_argument("bad input");} catch(const std::invalid_argument&){}')
add('RUN_MISSING_FILE_C',stage='runtime_error',before_bad='#include <stdio.h>',bad='FILE *f=fopen("absent_{v}.txt","rb"); if(!f) return 11; fclose(f);',good='FILE *f=tmpfile(); if(f) fclose(f);')
add('RUN_WRONG_LOGIC_C',stage='runtime_error',bad='if(seed_{v}+1 != seed_{v}) return 12;',good='if(seed_{v}+1 == seed_{v}) return 12;')
add('RUN_WRONG_LOGIC_CPP',lang='cpp',stage='runtime_error',bad='if(seed_{v}+1 != seed_{v}) return 12;',good='if(seed_{v}+1 == seed_{v}) return 12;')
add('RUN_MISSING_ENV_C',stage='runtime_error',before_bad='#include <stdlib.h>',bad='if(getenv("PGR_UNSET_{v}" ) == NULL) return 14;',good='if(getenv("PGR_UNSET_{v}" ) != NULL) return 14;')
add('RUN_INPUT_PARSE_C',stage='runtime_error',before_bad='#include <stdlib.h>',bad='char *end=0; const char *s="bad"; (void)strtol(s,&end,10); if(end==s) return 15;',good='char *end=0; const char *s="123"; (void)strtol(s,&end,10); if(end==s) return 15;')
add('RUN_DIVISION_GUARD_C',stage='runtime_error',bad='int denom=0; if(denom==0) return 16; return seed_{v}/denom;',good='int denom=1; if(denom==0) return 16; return (seed_{v}/denom) >= 0 ? 0 : 1;')

for code, mode in [
 ('CMAKE_UNKNOWN_COMMAND','unknown_command'), ('CMAKE_MISSING_SOURCE','missing_source'),
 ('CMAKE_MISSING_PACKAGE','missing_package'), ('CMAKE_INCLUDE_ABSENT','missing_include'),
 ('CMAKE_SUBDIRECTORY_ABSENT','missing_subdir'), ('CMAKE_DUPLICATE_TARGET','duplicate_target'),
 ('CMAKE_UNKNOWN_TARGET_LINK','target_link'), ('CMAKE_UNSUPPORTED_VERSION','future_version'),
 ('CMAKE_INVALID_LANGUAGE','invalid_language'),('CMAKE_BAD_ALIAS','bad_alias')]:
    add(code,lang='c',stage='configure_error',mode=mode)

BASE_C = '#include <stddef.h>\n#include <stdint.h>\n'
BASE_CPP = '#include <cstddef>\n#include <cstdint>\n'

def fmt(s, v, n):
    return s.replace('{v}',v).replace('{n}',str(n))

def main_code(f, v, n, bad):
    header=BASE_C if f.lang=='c' else BASE_CPP
    before=fmt(f.before_bad if bad else f.before_good,v,n)
    body=fmt(f.bad if bad else f.good,v,n)
    entry='int main(void)' if f.lang=='c' else 'int main()'
    return header+before+'\n'+entry+' {\n  int seed_'+v+' = '+str(n)+';\n  (void)seed_'+v+';\n  '+body+'\n  return 0;\n}\n'

def custom_link(f, v, n, bad):
    ext='.c' if f.lang=='c' else '.cpp'
    m=f.mode
    h='extern "C" ' if f.lang=='cpp' else ''
    if m=='undefined':
        decl='int helper_'+v+'(int);'
        call='int main(){return helper_'+v+'(1)==2 ? 0:1;}'
        return {'main'+ext:decl+'\n'+call+'\n', **({} if bad else {'helper'+ext:'int helper_'+v+'(int a){return a+1;}\n'})}
    if m=='no_main':
        return {'main'+ext: ('int entry_'+v+'(){return 0;}\n' if bad else 'int main(){return 0;}\n')}
    if m=='two_mains':
        return {'main'+ext:'int main(){return 0;}\n', 'extra'+ext: ('int main(){return 0;}\n' if bad else 'int helper_'+v+'(){return 0;}\n')}
    if m=='two_globals':
        return {'main'+ext:'int global_'+v+'=1; int main(){return global_'+v+'-1;}\n', 'extra'+ext: ('int global_'+v+'=2;\n' if bad else 'extern int global_'+v+';\n')}
    if m=='missing_tu':
        return {'main'+ext:'int helper_'+v+'(); int main(){return helper_'+v+'();}\n', **({} if bad else {'helper'+ext:'int helper_'+v+'(){return 0;}\n'})}
    if m=='static_member':
        decl='struct Item_'+v+'{static int value;};\n'
        return {'main'+ext: decl+ ('' if bad else 'int Item_'+v+'::value=0;\n')+'int main(){return Item_'+v+'::value;}\n'}
    if m=='virtual':
        decl='struct Item_'+v+'{virtual int value();};\n'
        return {'main'+ext: decl+ ('' if bad else 'int Item_'+v+'::value(){return 0;}\n')+'int main(){Item_'+v+' x; return x.value();}\n'}
    if m=='abi':
        return {'main.cpp':'extern "C" int helper_'+v+'(); int main(){return helper_'+v+'();}\n', 'extra.cpp':('int helper_'+v+'(){return 0;}\n' if bad else 'extern "C" int helper_'+v+'(){return 0;}\n')}
    if m=='library':
        return {'main'+ext:'int main(){return 0;}\n'}
    if m=='template':
        return {'main.cpp':'template<class T> int helper_'+v+'(T);\n'+('' if bad else 'template<class T> int helper_'+v+'(T){return 0;}\n')+'int main(){return helper_'+v+'(3);}\n'}
    raise ValueError(m)

def custom_cmake(mode, v, bad):
    standard='cmake_minimum_required(VERSION 3.16)\nproject(Fixture LANGUAGES C)\nadd_executable(fixture main.c)\n'
    if not bad:return standard
    if mode=='unknown_command':return standard+'pgr_made_up_command_'+v+'()\n'
    if mode=='missing_source':return standard.replace('main.c','gone_'+v+'.c')
    if mode=='missing_package':return standard+'find_package(PgrImaginary_'+v+' REQUIRED)\n'
    if mode=='missing_include':return standard+'include("${CMAKE_CURRENT_SOURCE_DIR}/no_file_'+v+'.cmake")\n'
    if mode=='missing_subdir':return standard+'add_subdirectory(absent_'+v+')\n'
    if mode=='duplicate_target':return standard+'add_executable(fixture main.c)\n'
    if mode=='target_link':return standard+'target_link_libraries(nonexistent_'+v+' PRIVATE m)\n'
    if mode=='future_version':return standard.replace('3.16','99.0')
    if mode=='invalid_language':return standard.replace('LANGUAGES C','LANGUAGES PGR_'+v)
    if mode=='bad_alias':return standard+'add_library(Foo_'+v+' ALIAS missing_'+v+')\n'
    raise ValueError(mode)

def contents(f,v,n,bad):
    if f.stage=='link_error':
        files=custom_link(f,v,n,bad)
        if f.lang=='c':
            files={name:re.sub(r'\b(int\s+(?:main|helper_'+v+'|entry_'+v+r'))\(\)',r'\1(void)',content) for name,content in files.items()}
    else:files={('main.c' if f.lang=='c' else 'main.cpp'):main_code(f,v,n,bad)}
    if f.stage=='configure_error':files['CMakeLists.txt']=custom_cmake(f.mode,v,bad)
    else:
        paths=' '.join(name for name in sorted(files) if name.endswith(('.c','.cpp')))
        files['CMakeLists.txt']=('cmake_minimum_required(VERSION 3.16)\nproject(Fixture LANGUAGES '+('C' if f.lang=='c' else 'CXX')+')\n'
          +'add_executable(fixture '+paths+')\n')
        if f.mode=='library' and bad:
            files['CMakeLists.txt']+='target_link_libraries(fixture PRIVATE pgr_absent_'+v+')\n'
    return files

def stable_split(group_key):
    family, bucket = group_key.split('/')
    # Every family contributes 6 train, 1 val, 1 test blocks of four pairs
    # when copies is a multiple of 32. This prevents family loss in validation.
    rotation=int(hashlib.sha256(('pgr-rotation-v3:'+family).encode()).hexdigest()[:8],16)%8
    slot=(int(bucket)+rotation)%8
    return 'train' if slot<6 else ('val' if slot==6 else 'test')

def generate(out, copies, seed):
    if out.exists():
        raise SystemExit('Output directory already exists; remove it explicitly: '+str(out))
    out.mkdir(parents=True)
    from layout import create_layout
    create_layout(out)
    rng=random.Random(seed)
    rows=[]
    order=list(range(len(F)*copies*2))
    rng.shuffle(order)
    used=set()
    for fi,f in enumerate(F):
        for variant in range(copies):
            n=1 + ((variant*17+fi*11)%91)
            v='v'+hashlib.blake2b(f'{fi}/{variant}/{seed}'.encode(),digest_size=4).hexdigest()
            pair_id=hashlib.sha256(f'{seed}-{f.code}-{variant}'.encode()).hexdigest()[:16]
            split=stable_split(f'{f.code}/{variant//4}')
            for bad in (True,False):
                token=hashlib.blake2b(f'{pair_id}-{bad}-{seed}'.encode(),digest_size=8).hexdigest()
                pid='p_'+token
                if pid in used:raise RuntimeError('hash collision')
                used.add(pid)
                path=out/'projects'/pid
                path.mkdir()
                files=contents(f,v,n,bad)
                for name,value in files.items():
                    (path/name).write_text(value,encoding='utf-8',newline='\n')
                row={'project_id':pid,'pair_id':pair_id,'split':split,'language':f.lang,'fault_code':f.code if bad else 'CLEAN',
                    'expected_stage':f.stage if bad else 'success','expected_ok':not bad,'variant':variant,
                    'sources':sorted(x for x in files if x.endswith(('.c','.cpp'))),
                    'build_system':'cmake' if f.stage=='configure_error' else 'direct+cmake',
                    'relative_path':'projects/'+pid,'toolchain':'GCC/Clang C11/C++17','generator_seed':seed,
                    'defect_mode':f.mode if bad else 'none'}
                rows.append(row)
    rng.shuffle(rows)
    def dump(path,data):
        with path.open('w',encoding='utf-8',newline='\n') as h:
            for x in data:h.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
    dump(out/'catalog'/'projects.jsonl',rows)
    for split in ('train','val','test'):
        dump(out/'splits'/f'{split}.jsonl',(r for r in rows if r['split']==split))
    with (out/'catalog'/'projects.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={'total':len(rows),'clean':sum(r['expected_ok'] for r in rows),
           'faulty':sum(not r['expected_ok'] for r in rows),'fault_families':len(F),
           'pairs':len(rows)//2,'copies_per_family':copies,
           'splits':{s:sum(r['split']==s for r in rows) for s in ('train','val','test')},
           'stages':{s:sum(r['expected_stage']==s for r in rows) for s in ('success','compile_error','link_error','configure_error','runtime_error')},
           'language':{s:sum(r['language']==s for r in rows) for s in ('c','cpp')},
           'seed':seed}
    (out/'catalog'/'summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (out/'catalog'/'fault_taxonomy.json').write_text(json.dumps(
        [{'code':f.code,'language':f.lang,'stage':f.stage,'mode':f.mode} for f in F],indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return stats

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out',type=Path,required=True,help='new data root; must not exist')
    a.add_argument('--copies',type=int,default=32,help='clean/defective pairs per error family')
    a.add_argument('--seed',type=int,default=20261028)
    args=a.parse_args()
    if not 1<=args.copies<=50000:raise SystemExit('copies out of range')
    print(json.dumps(generate(args.out,args.copies,args.seed),ensure_ascii=False,indent=2))
