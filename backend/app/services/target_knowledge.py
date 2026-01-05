from typing import Dict, Any

class TargetKnowledgeService:
    def __init__(self):
        self.knowledge_base = {
            # Inflammation
            "TNF": {
                "category": "Inflammation",
                "full_name": "Tumor Necrosis Factor",
                "description": "The 'Fire Alarm' of inflammation. Major driver of autoimmune disease.",
                "citation": "Liu, T., et al. (2017). Signal Transduct Target Ther.",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6159811/"
            },
            "IL6": {
                "category": "Inflammation",
                "full_name": "Interleukin-6",
                "description": "A 'Fever Signal' cytokine. Regulates chronic inflammation.",
                "citation": "Tanaka, T., et al. (2014). Cold Spring Harb Perspect Biol.",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4109575/"
            },
            "NFKB1": {
                "category": "Inflammation",
                "full_name": "Nuclear Factor Kappa B",
                "description": "The 'General'. A transcription factor that turns on inflammatory genes.",
                "citation": "Liu, T., et al. (2017). Signal Transduct Target Ther.",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6159811/"
            },
            
            # Oncology
            "TP53": {
                "category": "Cell Survival & Cancer",
                "full_name": "Tumor Protein p53",
                "description": "'Guardian of the Genome'. Stops cells from dividing if DNA is damaged.",
                "citation": "Aubrey, B.J., et al. (2018). Cell Death Differ.",
                "url": "https://www.nature.com/articles/cdd2017169"
            },
            "BCL2": {
                "category": "Cell Survival & Cancer",
                "full_name": "B-cell Lymphoma 2",
                "description": "'Survival Switch'. Prevents cell death. Overactive in many cancers.",
                "citation": "Warren, C.F.A., et al. (2019). Cell Death Dis.",
                "url": "https://www.nature.com/articles/s41419-019-1407-6"
            },
            "MMP9": {
                "category": "Tissue & Metastasis",
                "full_name": "Matrix Metalloproteinase-9",
                "description": "Enzyme that breaks down tissue, facilitating tumor invasion.",
                "citation": "Gialeli, C., et al. (2011). FEBS J.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/21251255/"
            },

            # Lipid & Cholesterol
            "HMGCR": {
                "category": "Cholesterol & Lipids",
                "full_name": "HMG-CoA Reductase",
                "description": "Rate-limiting enzyme of cholesterol synthesis. Primary target of Statins.",
                "citation": "Sharpe, L.J., et al. (2013). Nat Rev Mol Cell Biol.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/23881267/"
            },
            "SQLE": {
                "category": "Cholesterol & Lipids",
                "full_name": "Squalene Epoxidase",
                "description": "Key flux-controlling enzyme in cholesterol biosynthesis.",
                "citation": "Chua, N.K., et al. (2020). J Lipid Res.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/32680872/"
            },
            "MVD": {
                "category": "Cholesterol & Lipids",
                "full_name": "Mevalonate Diphosphate Decarboxylase",
                "description": "Critical enzyme in the mevalonate pathway for isoprenoid synthesis.",
                "citation": "Hogenboom, S., et al. (2004). Mol Genet Metab.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/14728989/"
            },

            # Neurology
            "MAPK1": {
                "category": "Brain & Plasticity",
                "full_name": "Mitogen-Activated Protein Kinase 1",
                "description": "Critical for forming new memories (Long Term Potentiation).",
                "citation": "Peng, S., et al. (2010). Int J Mol Sci.",
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2821004/"
            },
            "FAAH": {
                "category": "Anxiety & Stress",
                "full_name": "Fatty Acid Amide Hydrolase",
                "description": "Breaks down Anandamide ('Bliss Molecule'). Inhibiting it reduces anxiety.",
                "citation": "Gaetani, S., et al. (2009). CNS Neurol Disord Drug Targets.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/19796030/"
            },
            
            # Metabolism
            "CYP3A4": {
                "category": "Metabolism",
                "full_name": "Cytochrome P450 3A4",
                "description": "The liver's main detox engine. Metabolizes ~50% of drugs.",
                "citation": "Zanger, U.M., et al. (2013). Pharmacol Ther.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/23353742/"
            },
            "UGT1A1": {
                "category": "Metabolism",
                "full_name": "UDP-glucuronosyltransferase 1-1",
                "description": "Tags toxins and bilirubin for excretion.",
                "citation": "Rowland, A., et al. (2013). Expert Opin Drug Metab Toxicol.",
                "url": "https://pubmed.ncbi.nlm.nih.gov/23215858/"
            }
        }

    def enrich_target(self, gene_symbol: str) -> Dict[str, Any]:
        """Returns the enrichment dictionary for a gene, or None."""
        return self.knowledge_base.get(gene_symbol)
