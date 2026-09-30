"""
NERVA-ML: Anatomical Brain Region Ontology & Hierarchy
Maps 48 authentic human brain structures with hierarchical classifications
and established clinical vulnerability associations.
"""

from typing import Dict, List, Any

# 48 Standardized Anatomical Brain Regions across 6 major anatomical divisions
BRAIN_REGIONS: Dict[str, Dict[str, Any]] = {
    # --- BASAL GANGLIA (Primary vulnerability in Huntington's & Parkinson's) ---
    "Substantia Nigra pars compacta": {"division": "Brainstem/Mesencephalon", "abbreviation": "SNc", "system": "Dopaminergic"},
    "Substantia Nigra pars reticulata": {"division": "Brainstem/Mesencephalon", "abbreviation": "SNr", "system": "GABAergic"},
    "Caudate Nucleus": {"division": "Basal Ganglia", "abbreviation": "CAU", "system": "Striatal"},
    "Putamen": {"division": "Basal Ganglia", "abbreviation": "PUT", "system": "Striatal"},
    "Globus Pallidus internal": {"division": "Basal Ganglia", "abbreviation": "GPi", "system": "Pallidal"},
    "Globus Pallidus external": {"division": "Basal Ganglia", "abbreviation": "GPe", "system": "Pallidal"},
    "Nucleus Accumbens": {"division": "Basal Ganglia", "abbreviation": "NAc", "system": "Ventral Striatum"},
    "Subthalamic Nucleus": {"division": "Diencephalon", "abbreviation": "STN", "system": "Glutamatergic"},

    # --- LIMBIC SYSTEM & MEDIAL TEMPORAL (Primary vulnerability in Alzheimer's) ---
    "Hippocampus CA1": {"division": "Limbic System", "abbreviation": "CA1", "system": "Hippocampal"},
    "Hippocampus CA2": {"division": "Limbic System", "abbreviation": "CA2", "system": "Hippocampal"},
    "Hippocampus CA3": {"division": "Limbic System", "abbreviation": "CA3", "system": "Hippocampal"},
    "Hippocampus CA4": {"division": "Limbic System", "abbreviation": "CA4", "system": "Hippocampal"},
    "Dentate Gyrus": {"division": "Limbic System", "abbreviation": "DG", "system": "Hippocampal"},
    "Subiculum": {"division": "Limbic System", "abbreviation": "Sub", "system": "Parahippocampal"},
    "Entorhinal Cortex": {"division": "Cerebral Cortex", "abbreviation": "EC", "system": "Parahippocampal"},
    "Parahippocampal Gyrus": {"division": "Cerebral Cortex", "abbreviation": "PHG", "system": "Parahippocampal"},
    "Basolateral Amygdala": {"division": "Limbic System", "abbreviation": "BLA", "system": "Amygdalar"},
    "Corticomedial Amygdala": {"division": "Limbic System", "abbreviation": "CMA", "system": "Amygdalar"},
    "Anterior Cingulate Cortex": {"division": "Cerebral Cortex", "abbreviation": "ACC", "system": "Cingulate"},
    "Posterior Cingulate Cortex": {"division": "Cerebral Cortex", "abbreviation": "PCC", "system": "Cingulate"},

    # --- CEREBRAL CORTEX (Motor, Sensory, Association) ---
    "Primary Motor Cortex": {"division": "Cerebral Cortex", "abbreviation": "M1", "system": "Motor"},
    "Premotor Cortex": {"division": "Cerebral Cortex", "abbreviation": "PMC", "system": "Motor"},
    "Supplementary Motor Area": {"division": "Cerebral Cortex", "abbreviation": "SMA", "system": "Motor"},
    "Primary Somatosensory Cortex": {"division": "Cerebral Cortex", "abbreviation": "S1", "system": "Sensory"},
    "Dorsolateral Prefrontal Cortex": {"division": "Cerebral Cortex", "abbreviation": "DLPFC", "system": "Executive"},
    "Ventrolateral Prefrontal Cortex": {"division": "Cerebral Cortex", "abbreviation": "VLPFC", "system": "Executive"},
    "Orbitofrontal Cortex": {"division": "Cerebral Cortex", "abbreviation": "OFC", "system": "Frontal"},
    "Frontal Pole": {"division": "Cerebral Cortex", "abbreviation": "FP", "system": "Frontal"},
    "Superior Temporal Gyrus": {"division": "Cerebral Cortex", "abbreviation": "STG", "system": "Auditory/Temporal"},
    "Middle Temporal Gyrus": {"division": "Cerebral Cortex", "abbreviation": "MTG", "system": "Association"},
    "Inferior Temporal Gyrus": {"division": "Cerebral Cortex", "abbreviation": "ITG", "system": "Visual Association"},
    "Fusiform Gyrus": {"division": "Cerebral Cortex", "abbreviation": "FFG", "system": "Ventral Stream"},
    "Superior Parietal Lobule": {"division": "Cerebral Cortex", "abbreviation": "SPL", "system": "Parietal"},
    "Inferior Parietal Lobule": {"division": "Cerebral Cortex", "abbreviation": "IPL", "system": "Parietal"},
    "Precuneus": {"division": "Cerebral Cortex", "abbreviation": "PCu", "system": "Default Mode"},
    "Primary Visual Cortex (V1)": {"division": "Cerebral Cortex", "abbreviation": "V1", "system": "Visual"},
    "Insular Cortex": {"division": "Cerebral Cortex", "abbreviation": "INS", "system": "Salience"},

    # --- DIENCEPHALON & THALAMUS ---
    "Anterior Thalamic Nucleus": {"division": "Diencephalon", "abbreviation": "ATN", "system": "Thalamocortical"},
    "Ventral Lateral Thalamus": {"division": "Diencephalon", "abbreviation": "VL", "system": "Motor Relay"},
    "Pulvinar Thalamus": {"division": "Diencephalon", "abbreviation": "Pul", "system": "Visual Relay"},
    "Hypothalamus": {"division": "Diencephalon", "abbreviation": "HYP", "system": "Neuroendocrine"},

    # --- BRAINSTEM & MONOAMINERGIC NUCLEI ---
    "Locus Coeruleus": {"division": "Brainstem/Pons", "abbreviation": "LC", "system": "Noradrenergic"},
    "Raphe Nuclei": {"division": "Brainstem/Midbrain", "abbreviation": "RN", "system": "Serotonergic"},
    "Pedunculopontine Nucleus": {"division": "Brainstem/Pons", "abbreviation": "PPN", "system": "Cholinergic"},
    "Red Nucleus": {"division": "Brainstem/Midbrain", "abbreviation": "RNuc", "system": "Motor"},

    # --- CEREBELLUM ---
    "Cerebellar Cortex Anterior": {"division": "Cerebellum", "abbreviation": "CB-Ant", "system": "Sensorimotor"},
    "Cerebellar Cortex Posterior": {"division": "Cerebellum", "abbreviation": "CB-Post", "system": "Cognitive"},
    "Dentate Nucleus of Cerebellum": {"division": "Cerebellum", "abbreviation": "DN", "system": "Cerebellar Deep"},
}

def get_region_names() -> List[str]:
    """Returns the ordered list of 48 standardized brain regions."""
    return list(BRAIN_REGIONS.keys())

def get_region_divisions() -> Dict[str, str]:
    """Returns mapping from region name to anatomical division."""
    return {name: info["division"] for name, info in BRAIN_REGIONS.items()}

# Ground-truth clinical vulnerability definitions from neuropathology literature
GROUND_TRUTH_VULNERABILITY = {
    "Parkinson's Disease": [
        "Substantia Nigra pars compacta",
        "Substantia Nigra pars reticulata",
        "Caudate Nucleus",
        "Putamen",
        "Locus Coeruleus",
        "Primary Motor Cortex",
        "Supplementary Motor Area",
        "Pedunculopontine Nucleus"
    ],
    "Alzheimer's Disease": [
        "Hippocampus CA1",
        "Hippocampus CA2",
        "Hippocampus CA3",
        "Hippocampus CA4",
        "Dentate Gyrus",
        "Subiculum",
        "Entorhinal Cortex",
        "Parahippocampal Gyrus",
        "Posterior Cingulate Cortex",
        "Precuneus",
        "Superior Temporal Gyrus",
        "Dorsolateral Prefrontal Cortex"
    ],
    "Huntington's Disease": [
        "Caudate Nucleus",
        "Putamen",
        "Globus Pallidus internal",
        "Globus Pallidus external",
        "Nucleus Accumbens",
        "Subthalamic Nucleus",
        "Primary Motor Cortex"
    ]
}
