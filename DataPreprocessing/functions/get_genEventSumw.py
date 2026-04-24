import ROOT
def get_genEventSumw(input_file, maxEntriesPerSample=None):
   '''
      Util function to get the sum of weights per event.
      Returns the sum of weights, similarly to what we
      stored in Counters->GetBinContent(40) in the miniAODs.
   '''
   f = ROOT.TFile.Open(input_file)


   runs  = f['Runs']
   event = f['Events']
   nRuns = runs.GetEntries()
   nEntries = event.GetEntries()


   iRun = 0
   genEventCount = 0
   genEventSumw = 0.


   while iRun < nRuns and runs.GetEntry(iRun) :
       genEventCount += runs.genEventCount
       genEventSumw += runs.genEventSumw
       iRun +=1
   print ("gen=", genEventCount, "sumw=", genEventSumw)


   if maxEntriesPerSample is not None:
       print(f"Scaling to {maxEntriesPerSample} entries")
       if nEntries>maxEntriesPerSample :
           genEventSumw = genEventSumw*maxEntriesPerSample/nEntries
           nEntries=maxEntriesPerSample
       print("    scaled to:", nEntries, "sumw=", genEventSumw)


   return genEventSumw