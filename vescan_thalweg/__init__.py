"""vescan_thalweg: VeScan with the centerlines of Mike Halle's thalweg, from the haversack ranked store
(no VMTK). Only the centerlines stage differs from vescan_field: it calls thalweg.

Generic part:
    config        typed JSON config (batch + pipeline, one section per stage)
    log           logging: console + one file per patient
    margin        a structure's (union of classes) margin from the store, on its own grid, for thalweg
    field         CT grid geometry of the masks
    io            JSON, images, markups
    surface       VTK surface utilities shared by the stages
    group_tree    the group graph of the topology JSON
    centerline_distance  nearest centerline segment per point (vtkvmtkPolyBallLine's metric)
    lobes         lobe order, mask names, sides
    orchestrator  patient -> structures -> margin -> enabled stages

Stages (vescan_thalweg/stages/<name>/, see vescan_thalweg/stages/__init__.py).
"""
