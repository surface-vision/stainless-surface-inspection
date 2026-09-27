import json
R=167407; V_line=0.4e6; V_all=2565902
cap_lo,cap_hi=2.5,4.2; cap=(cap_lo+cap_hi)/2; opex=0.5; rec=0.30
def pool(v,d,s): return v*d*s*R/1e7
grid={}
for d in (0.005,0.01,0.02):
  for s in (0.10,0.15,0.20):
    grid[f"{d}|{s}"]=round(pool(V_line,d,s),1)
def payback(d=0.01,s=0.10,r=rec,c=cap,v=V_line,o=opex):
  net=pool(v,d,s)*r-o
  return c/net if net>0 else 99
base=dict(d=0.01,s=0.15)
out={"grid_line_cr":grid,"capex_mid":cap}
out["mid_pool"]=pool(V_line,0.01,0.15); out["mid_recovered"]=out["mid_pool"]*rec
out["mid_payback"]=payback()
out["payback_1pct_10"]=payback(s=0.10)
out["payback_hi"]=payback(d=0.02,s=0.20); out["payback_lo"]=payback(d=0.005,s=0.10)
comm=19.1  # USD 1-3M midpoint 2M * 95.8
out["commercial_capex_cr"]=2e6*95.8/1e7
out["comm_payback_mid"]=payback(c=out["commercial_capex_cr"])
# tornado on mid case
t={}
t["Downgrade rate 0.5-2%"]=(payback(d=0.02),payback(d=0.005))
t["Discount 10-20%"]=(payback(s=0.20),payback(s=0.10))
t["Recovery 15-45%"]=(payback(r=0.45),payback(r=0.15))
t["Capex INR 2.5-4.2 cr"]=(payback(c=2.5),payback(c=4.2))
t["Line volume 0.3-0.5 Mt"]=(payback(v=0.5e6),payback(v=0.3e6))
out["tornado"]={k:(round(a,2),round(b,2)) for k,(a,b) in t.items()}
# breakeven d*s for 2-yr payback
need=(cap/2+opex)/rec*1e7
out["breakeven_ds_2yr"]=need/(V_line*R)
# NPV 5yr 12% mid
npv=-cap+sum((pool(V_line,0.01,0.10)*rec-opex)/1.12**y for y in range(1,6))
out["npv5_mid"]=npv
out["fleet_pool_mid"]=pool(V_all,0.01,0.15); out["fleet_recovered"]=out["fleet_pool_mid"]*rec
out["fleet_pool_1_10"]=pool(V_all,0.01,0.10)
out["lines_equiv"]=V_all/V_line
# nickel
for ni in (0.08,0.105): out[f"ni_{ni}"]=ni*16402*95.8
print(json.dumps(out,indent=1))
