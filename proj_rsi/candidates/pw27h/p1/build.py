import sys, time, importlib.util
spec = importlib.util.spec_from_file_location("submission", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
t = time.time(); mod._gvext(); print("gv built %.0fs, %d instances" % (time.time() - t, len(mod.GVI)), flush=True)
t = time.time(); mod._ext(); print("ext built %.0fs" % (time.time() - t), flush=True)
