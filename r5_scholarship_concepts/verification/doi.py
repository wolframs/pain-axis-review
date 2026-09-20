import urllib.request, json, time
dois=["10.1016/j.applanim.2009.03.013","10.1613/jair.1.17310","10.1007/s44204-025-00246-2","10.4324/9781003674573",
"10.1016/j.applanim.2005.06.018","10.1016/j.isci.2024.110440","10.1037/0033-295X.115.1.186","10.1016/S0003-3472(83)80026-8",
"10.1002/14651858.CD008659.pub3","10.1016/S0304-3959(00)00413-9","10.18653/v1/2020.acl-main.372","10.1142/S270507852150003X",
"10.18653/v1/2024.findings-naacl.130","10.18653/v1/2024.acl-long.828","10.1017/CBO9780511975950","10.48550/arXiv.2407.12404",
"10.48550/arXiv.2309.16042","10.1136/vr.146.11.307"]
for d in dois:
    req=urllib.request.Request("https://doi.org/"+d, headers={"Accept":"application/vnd.citationstyles.csl+json","User-Agent":"ref-check"})
    try:
        j=json.loads(urllib.request.urlopen(req,timeout=30).read())
    except Exception as e:
        print(d,"ERR",e); time.sleep(1); continue
    au=j.get('author',[])
    names=", ".join((a.get('family') or a.get('literal','?')) for a in au[:8])
    dt=j.get('issued',{}).get('date-parts',[[None]])[0]
    print(f"{d}\n   title: {j.get('title')}\n   authors: {names}\n   container: {j.get('container-title')} vol={j.get('volume')} issue={j.get('issue')} page={j.get('page')} year={dt}\n")
    time.sleep(1)
