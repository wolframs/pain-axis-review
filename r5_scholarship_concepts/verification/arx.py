import urllib.request, xml.etree.ElementTree as ET, time, sys
ids = ["2406.11717","2508.16560","2507.21509","2304.11111","2509.04781","2101.00027","2609.07037","2411.02432","2408.05147","2411.00986","2601.10387","2509.07961","2407.12404","2308.10248","2608.26178","2608.08159","2309.16042","2309.03882","2308.08708","2604.07729"]
ns={'a':'http://www.w3.org/2005/Atom'}
for i in ids:
    u=f"http://export.arxiv.org/api/query?id_list={i}&max_results=1"
    try:
        x=urllib.request.urlopen(u,timeout=30).read()
    except Exception as e:
        print(i,"FETCH-ERR",e); continue
    r=ET.fromstring(x)
    e=r.find('a:entry',ns)
    if e is None: print(i,"NOT FOUND"); continue
    t=' '.join(e.find('a:title',ns).text.split())
    au=[a.find('a:name',ns).text for a in e.findall('a:author',ns)]
    pub=e.find('a:published',ns).text[:10]; upd=e.find('a:updated',ns).text[:10]
    jr=e.find('a:journal_ref',ns)
    print(f"{i} | {t}")
    print(f"     authors({len(au)}): {', '.join(au[:6])}{' ...' if len(au)>6 else ''}")
    print(f"     v1={pub} latest={upd} journal_ref={jr.text if jr is not None else '-'}")
    time.sleep(3)
