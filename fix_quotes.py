with open("templates/discover.html", "r") as f:
    r = f.read()

r = r.replace("onclick=\\\"cancelJob(\\\\'\" + e.jobId + \"\\\\')\\\"", "onclick=\\\"cancelJob('\" + e.jobId + \"')\\\"")
r = r.replace("cancelJob(\\\\'\" + e.jobId + \"\\\\')", "cancelJob('\" + e.jobId + \"')")
r = r.replace("cancelJob(\\'\" + e.jobId + \"\\')", "cancelJob('\" + e.jobId + \"')")
with open("templates/discover.html", "w") as f:
    f.write(r)
