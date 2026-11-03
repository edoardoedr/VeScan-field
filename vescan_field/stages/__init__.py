"""Pipeline stages, one per subfolder, run in orchestrator.STAGES order.
Each exposes run_patient(pctx, cfg) (once per patient) and/or run(ctx, cfg) (per structure),
with cfg its own pipeline.<name> config section; the returned dict goes into the summary.

    surfaces        surfaces from the masks: lobes + fissures, vessels, 01_preprocessed.vtk
    centerlines     centerlines, graph, groups and the 05_ files for VeScan
    anatomy         lobe reachability and anatomical segments on the centerline (09_, 10_)
    surface_labels  centerline labels onto the whole surface (08_)
    clip            last: centerline cut after N generations and clip of the labelled surface (06_, 07_)
"""
