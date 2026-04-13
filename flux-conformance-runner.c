/*
 * flux-conformance-runner.c — C11 conformance runner for FLUX ISA
 * Matches SuperInstance/flux-conformance conformance_core.py exactly.
 * Build: gcc -O2 -Wall -o runner flux-conformance-runner.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MAX_STACK 1024
#define MAX_CODE  8192
#define MEM_SIZE  65536

typedef struct {
    double s[MAX_STACK]; int sp; // polymorphic stack (int or float)
    int is_float[MAX_STACK]; // 1=float, 0=int
    float fs[MAX_STACK]; int fsp;
    uint8_t bc[MAX_CODE]; int len, pc;
    int running, halted, steps;
    int call_stack[256]; int csp;
    uint8_t mem[MEM_SIZE];
    float conf;
    int32_t sig[256][64]; int sig_len[256]; int sig_head[256];
    int err; char estr[64];
    // Flags
    int fz, fsign, fc, fo;
} VM;

static void vm_init(VM* v){memset(v,0,sizeof(*v));v->conf=1.0f;}
static void push_i(VM*v,int32_t x){if(v->sp<MAX_STACK){v->s[v->sp]=(double)x;v->is_float[v->sp]=0;v->sp++;}}
static void push_f(VM*v,double x){if(v->sp<MAX_STACK){v->s[v->sp]=x;v->is_float[v->sp]=1;v->sp++;}}
static double pop_d(VM*v){return v->sp>0?v->s[--v->sp]:0;}
static int32_t pop_i(VM*v){return v->sp>0?(int32_t)v->s[--v->sp]:0;}
static void fpush(VM*v,float x){if(v->fsp<MAX_STACK)v->fs[v->fsp++]=x;}
static float fpop(VM*v){return v->fsp>0?v->fs[--v->fsp]:0;}
static uint8_t read_u8(VM*v){return v->bc[v->pc++];}
static int32_t read_i32(VM*v){int32_t r;memcpy(&r,&v->bc[v->pc],4);v->pc+=4;return r;}
static uint16_t read_u16(VM*v){uint16_t r;memcpy(&r,&v->bc[v->pc],2);v->pc+=2;return r;}
static void uf_arith(VM*v,int32_t r,int32_t a,int32_t b,int is_sub){
    v->fz=(r==0);v->fsign=(r<0);
    if(is_sub)v->fc=(a>=0&&b>=0&&a<b); else{uint32_t ua=a,ub=b;v->fc=(ua+ub)<ua;}
    v->fo=0; // simplified
}
static void uf_logic(VM*v,int32_t r){v->fz=(r==0);v->fsign=(r<0);v->fc=0;v->fo=0;}

// Python int(a/b) = truncation toward zero
static int32_t py_div(int32_t a,int32_t b){
    if(b==0)return 0;
    int32_t q=a/b;int32_t r=a%b;
    // C truncates toward zero already for int32
    // Python: int(a/b) uses float division then truncates
    // For negative: Python int(-7/2) = int(-3.5) = -3 (trunc toward zero)
    // C: -7/2 = -3 (same). But -7%2 = -1 in C, 1 in Python
    // Actually Python a%b has same sign as b. C has same sign as a.
    return q;
}
static int32_t py_mod(int32_t a,int32_t b){
    if(b==0)return 0;
    // Python: a%b has same sign as b
    int32_t r=a%b;
    if(r!=0&&(r>0)!=(b>0))r+=b;
    return r;
}

static void run(VM*v){
    v->running=1;
    while(v->running&&v->pc<v->len&&v->steps<1000000){
        uint8_t op=read_u8(v);v->steps++;
        int32_t a,b,r;float fa,fb;
        switch(op){
        case 0x00: v->halted=1;v->running=0;break; // HALT
        case 0x01: break; // NOP
        case 0x02: v->running=0;break; // BREAK
        case 0x10: b=pop_i(v);a=pop_i(v);r=a+b;uf_arith(v,r,a,b,0);push_i(v,r);break; // ADD
        case 0x11: b=pop_i(v);a=pop_i(v);r=a-b;uf_arith(v,r,a,b,1);push_i(v,r);break; // SUB
        case 0x12: b=pop_i(v);a=pop_i(v);r=a*b;uf_arith(v,r,a,b,0);push_i(v,r);break; // MUL
        case 0x13: b=pop_i(v);a=pop_i(v);if(!b){v->err=5;strcpy(v->estr,"DIV_ZERO");return;}r=py_div(a,b);uf_arith(v,r,a,b,0);push_i(v,r);break; // DIV
        case 0x14: b=pop_i(v);a=pop_i(v);if(!b){v->err=5;strcpy(v->estr,"MOD_ZERO");return;}r=py_mod(a,b);uf_arith(v,r,a,b,0);push_i(v,r);break; // MOD
        case 0x15: a=pop_i(v);r=-a;uf_arith(v,r,0,a,1);push_i(v,r);break; // NEG
        case 0x16: a=pop_i(v);r=a+1;uf_arith(v,r,a,1,0);push_i(v,r);break; // INC
        case 0x17: a=pop_i(v);r=a-1;uf_arith(v,r,a,1,1);push_i(v,r);break; // DEC
        case 0x20: b=pop_i(v);a=pop_i(v);r=(a==b)?1:0;uf_logic(v,r);push_i(v,r);break; // EQ
        case 0x21: b=pop_i(v);a=pop_i(v);r=(a!=b)?1:0;uf_logic(v,r);push_i(v,r);break; // NE
        case 0x22: b=pop_i(v);a=pop_i(v);r=(a<b)?1:0;uf_logic(v,r);push_i(v,r);break; // LT
        case 0x23: b=pop_i(v);a=pop_i(v);r=(a<=b)?1:0;uf_logic(v,r);push_i(v,r);break; // LE
        case 0x24: b=pop_i(v);a=pop_i(v);r=(a>b)?1:0;uf_logic(v,r);push_i(v,r);break; // GT
        case 0x25: b=pop_i(v);a=pop_i(v);r=(a>=b)?1:0;uf_logic(v,r);push_i(v,r);break; // GE
        case 0x30: b=pop_i(v);a=pop_i(v);r=(int32_t)((int32_t)a&(int32_t)b);uf_logic(v,r);push_i(v,r);break; // AND
        case 0x31: b=pop_i(v);a=pop_i(v);r=(int32_t)((int32_t)a|(int32_t)b);uf_logic(v,r);push_i(v,r);break; // OR
        case 0x32: b=pop_i(v);a=pop_i(v);r=(int32_t)((int32_t)a^(int32_t)b);uf_logic(v,r);push_i(v,r);break; // XOR
        case 0x33: a=pop_i(v);r=~a;uf_logic(v,r);push_i(v,r);break; // NOT
        case 0x34: b=pop_i(v);a=pop_i(v);r=(int32_t)((uint32_t)a<<(b&0x1F));uf_logic(v,r);push_i(v,r);break; // SHL
        case 0x35: b=pop_i(v);a=pop_i(v);r=a>>(b&0x1F);uf_logic(v,r);push_i(v,r);break; // SHR
        case 0x40: {uint16_t addr=read_u16(v);int32_t val;memcpy(&val,&v->mem[addr],4);push_i(v,val);break;} // LOAD addr
        case 0x41: {uint16_t addr=read_u16(v);int32_t val=pop_i(v);memcpy(&v->mem[addr],&val,4);break;} // STORE addr val
        case 0x43: {int32_t addr=pop_i(v);int32_t val;memcpy(&val,&v->mem[addr],4);push_i(v,val);break;} // PEEK
        case 0x44: {int32_t val=pop_i(v);int32_t addr=pop_i(v);memcpy(&v->mem[addr],&val,4);break;} // POKE val addr
        case 0x50: v->pc=read_u16(v);break; // JMP addr
        case 0x51: {uint16_t addr=read_u16(v);if(v->fz)v->pc=addr;break;} // JZ addr
        case 0x52: {uint16_t addr=read_u16(v);if(!v->fz)v->pc=addr;break;} // JNZ addr
        case 0x53: {uint16_t addr=read_u16(v);v->call_stack[v->csp++]=v->pc;v->pc=addr;break;} // CALL addr
        case 0x54: if(v->csp>0)v->pc=v->call_stack[--v->csp];break; // RET
        case 0x55: push_i(v,read_i32(v));break; // PUSH imm32
        case 0x56: pop_i(v);break; // POP
        case 0x60: {a=pop_i(v);push_i(v,a);push_i(v,a);break;} // DUP
        case 0x61: {a=pop_i(v);b=pop_i(v);push_i(v,a);push_i(v,b);break;} // SWAP
        case 0x62: {a=pop_i(v);b=pop_i(v);push_i(v,b);push_i(v,a);push_i(v,b);break;} // OVER
        case 0x63: // ROT [a b c] -> [b c a]
            if(v->sp>=3){double c=v->s[v->sp-3];v->s[v->sp-3]=v->s[v->sp-2];v->s[v->sp-2]=v->s[v->sp-1];v->s[v->sp-1]=c;}
            break;
        case 0x70: {double b=pop_d(v),a=pop_d(v);push_f(v,a+b);break;} // FADD
        case 0x71: {double b=pop_d(v),a=pop_d(v);push_f(v,a-b);break;} // FSUB
        case 0x72: {double b=pop_d(v),a=pop_d(v);push_f(v,a*b);break;} // FMUL
        case 0x73: {double b=pop_d(v),a=pop_d(v);if(!b){v->err=5;return;}push_f(v,a/b);break;} // FDIV
        case 0x80: push_f(v,(double)v->conf);break; // CONF_GET -> float
        case 0x81: {double val=pop_d(v);v->conf=val<0?0:val>1?1:val;break;} // CONF_SET -> pop, clamp to [0,1]
        case 0x82: {double b=pop_d(v);v->conf*=b;v->conf=v->conf<0?0:v->conf>1?1:v->conf;push_f(v,(double)v->conf);break;} // CONF_MUL
        case 0x90: {uint8_t ch=read_u8(v);int32_t val=pop_i(v);if(ch<256&&v->sig_len[ch]<64){v->sig[ch][(v->sig_head[ch]+v->sig_len[ch])%64]=val;v->sig_len[ch]++;}break;} // SIGNAL ch val
        case 0x91: {uint8_t ch=read_u8(v);int32_t val=pop_i(v);if(ch<256&&v->sig_len[ch]<64){v->sig[ch][(v->sig_head[ch]+v->sig_len[ch])%64]=val;v->sig_len[ch]++;}break;} // BROADCAST ch val
        case 0x92: {uint8_t ch=read_u8(v);if(ch<256&&v->sig_len[ch]>0){push_i(v,v->sig[ch][v->sig_head[ch]]);v->sig_head[ch]=(v->sig_head[ch]+1)%64;v->sig_len[ch]--;}else push_i(v,0);break;} // LISTEN ch
        default: break;
        }
    }
    if(!v->halted&&v->running){v->err=6;strcpy(v->estr,"MAX_STEPS");}
}

static int parse_hex(const char*hex,uint8_t*out,int max){
    int len=strlen(hex),i=0,o=0;
    while(i<len&&o<max){while(i<len&&hex[i]==' ')i++;if(i>=len)break;
        int hi=-1,lo=-1;char c=hex[i];
        if(c>='0'&&c<='9')hi=c-'0';else if(c>='a'&&c<='f')hi=c-'a'+10;else if(c>='A'&&c<='F')hi=c-'A'+10;
        i++;if(i<len){c=hex[i];if(c>='0'&&c<='9')lo=c-'0';else if(c>='a'&&c<='f')lo=c-'a'+10;else if(c>='A'&&c<='F')lo=c-'A'+10;i++;}
        if(hi>=0&&lo>=0)out[o++]=(hi<<4)|lo;
    }
    return o;
}

int main(void){
    char line[8192];int passed=0,failed=0,total=0;
    while(fgets(line,sizeof(line),stdin)){
        if(line[0]!='V')continue;
        char name[128]={0},hex[4096]={0},is[512]={0},es[512]={0},fs[16]={0},er[64]={0},desc[256]={0};
        char*p=line+2;int f;
        for(f=0;*p&&*p!='|'&&f<127;p++)name[f++]=*p;name[f]=0;if(*p)p++;
        for(f=0;*p&&*p!='|'&&f<4095;p++)hex[f++]=*p;hex[f]=0;if(*p)p++;
        for(f=0;*p&&*p!='|'&&f<511;p++)is[f++]=*p;is[f]=0;if(*p)p++;
        for(f=0;*p&&*p!='|'&&f<511;p++)es[f++]=*p;es[f]=0;if(*p)p++;
        for(f=0;*p&&*p!='|'&&f<15;p++)fs[f++]=*p;fs[f]=0;if(*p)p++;
        for(f=0;*p&&*p!='|'&&f<63;p++)er[f++]=*p;er[f]=0;if(*p)p++;
        for(f=0;*p&&*p!='\n'&&*p!='\r'&&f<255;p++)desc[f++]=*p;desc[f]=0;

        double ini_d[MAX_STACK],exp[MAX_STACK];int ic=0,ec=0;
        if(strlen(is)>0){char buf[512];strcpy(buf,is);char*tok=strtok(buf,",");while(tok&&ic<MAX_STACK){ini_d[ic++]=atof(tok);tok=strtok(NULL,",");}}
        if(strlen(es)>0){char buf[512];strcpy(buf,es);char*tok=strtok(buf,",");while(tok&&ec<MAX_STACK){exp[ec++]=atof(tok);tok=strtok(NULL,",");}}
        int eflags=atoi(fs);int expect_err=(strlen(er)>0);
        total++;

        VM vm;vm_init(&vm);
        for(int i=0;i<ic;i++)push_i(&vm,(int32_t)ini_d[i]); // first in list = bottom of stack
        vm.len=parse_hex(hex,vm.bc,MAX_CODE);vm.pc=0;
        run(&vm);

        int ok=1;
        if(expect_err){if(vm.err==0)ok=0;}
        else{
            if(vm.err!=0)ok=0;
            if(ok){for(int i=0;i<ec;i++){double got=vm.s[i];double want=exp[i];if(fabs(got-want)>0.0001){ok=0;break;}}}
            if(ok&&eflags>=0){
                int af=(vm.fz?1:0)|(vm.fsign?2:0)|(vm.fc?4:0)|(vm.fo?8:0);
                if(af!=eflags)ok=0;
            }
        }
        if(ok){passed++;printf("  ✅ %-50s %s\n",name,desc);}
        else{failed++;printf("  ❌ %-50s %s\n",name,desc);
            printf("      exp_stack=%d got=%d err=%d(%s)\n",ec,vm.sp,vm.err,vm.estr);
            printf("      flags: exp=%d got=%d\n",eflags,(vm.fz?1:0)|(vm.fsign?2:0)|(vm.fc?4:0));
        }
    }
    printf("\n═══════════════════════════════════════════\n");
    printf("  %d/%d passed (%d failed)\n",passed,total,failed);
    printf("═══════════════════════════════════════════\n");
    return failed>0?1:0;
}
