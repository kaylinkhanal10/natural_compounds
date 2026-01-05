from .neo4j_service import Neo4jService
from ..ml.world_model.infer import WorldModelInference

class ReasoningEngine:
    def __init__(self):
        self.db = Neo4jService()
        try:
            self.vae = WorldModelInference()
            print("ReasoningEngine: VAE World Model loaded.")
        except Exception as e:
            print(f"ReasoningEngine: Failed to load VAE. {e}")
            self.vae = None

    def analyze_combination(self, herb_ids: list):
        # 1. Shared Effects (Graph Logic)
        shared_effects = self.db.find_shared_effects(herb_ids)
        
        # 2. Interactions (Graph Logic)
        interactions = self.db.find_interactions(herb_ids)
        
        # 3. VAE Synergy Analysis (AI Logic)
        vae_analysis = None
        if self.vae and len(herb_ids) == 2:
            try:
                # Fetch compounds for the two herbs
                # We need a db method for this, or run a quick custom query here? 
                # Let's run a custom query via self.db.driver
                
                # Helper to get compounds
                def get_compounds(hid):
                    # Explicit Mapping for MVP Herbs (Scientific Name -> Database Latin Name)
                    HERB_NAME_MAP = {
                        "Curcuma longa": "Curcumae Longae Rhizoma",              # Turmeric
                        "Piper longum": "Piperis Longi Fructus",                 # Pippali
                        "Zingiber officinale": "Zingiberis Rhizoma",             # Ginger
                        "Phyllanthus emblica": "Phyllanthi Fructus",             # Amla
                        "Tinospora cordifolia": "Tinosporae Radix",              # Guduchi
                        "Glycyrrhiza glabra": "Glycyrrhizae Radix Et Rhizoma"    # Licorice
                    }
                    
                    q = """
                    MATCH (h:Herb {herbId: $hid})
                    // Resolving Name Mismatch
                    WITH h, $map[h.scientificName] as mapped_name
                    MATCH (hm:HerbMaterial) 
                    WHERE hm.latin_name = h.scientificName OR (mapped_name IS NOT NULL AND hm.latin_name = mapped_name)
                    MATCH (hm)-[:HAS_COMPOUND]->(c:Compound)
                    RETURN collect(c.inchikey) as inchikeys
                    """
                    with self.db.driver.session() as s:
                        res = s.run(q, hid=hid, map=HERB_NAME_MAP).single()
                        return res['inchikeys'] if res else []
                
                comps_a = get_compounds(herb_ids[0])
                comps_b = get_compounds(herb_ids[1])
                
                metric = self.vae.compare_herb_profiles(comps_a, comps_b)
                if metric:
                    vae_analysis = metric
            except Exception as e:
                print(f"VAE Analysis failed: {e}")
        
        # 4. Explanation Construction
        explanation = []
        if shared_effects:
            explanation.append("Mechanistic overlap found: The selected herbs share common biological effects.")
            for eff in shared_effects:
                names = ", ".join(eff['herb_names'])
                explanation.append(f"- {names} both contribute to {eff['e']['name']} ({eff['e']['category']}).")
        
        if interactions:
            explanation.append("\nNote on Interactions:")
            for inter in interactions:
                explanation.append(f"- {inter['description']}")
                
        if vae_analysis:
            explanation.append("\nAI World Model Analysis:")
            dist = vae_analysis['centroid_distance']
            if dist < 0.2:
                explanation.append(f"- High Redundancy (Distance {dist:.2f}): These herbs occupy very similar biological space.")
            elif dist > 0.5:
                explanation.append(f"- High Complementarity (Distance {dist:.2f}): These herbs cover distinct chemical-biological profiles, suggesting broad synergy.")
            else:
                explanation.append(f"- Moderate Overlap (Distance {dist:.2f}): Balanced profile.")

        # 5. Semantic Target Enrichment
        from .target_knowledge import TargetKnowledgeService
        tk = TargetKnowledgeService()
        
        # We need the list of shared proteins to enrich. 
        # Currently 'shared_effects' has some, but the Main Graph Query (in api) provides the full list.
        # However, let's look at 'shared_effects' (Graph Logic 1) -> it gives {e: {category, name}}.
        # This is strictly "Effect" nodes, not Protein nodes.
        # But we can try to find protein intersections here if we want?
        # Actually, best place is to return a helper object that the API can use, 
        # OR we just add a generic "Key Targets Identified" section if we can find them.
        
        # Let's run a quick query to get the top shared proteins strictly for the explanation text
        q_top_shared = """
        MATCH (h:Herb) WHERE h.herbId IN $herbs
        MATCH (hm:HerbMaterial) WHERE hm.latin_name = h.scientificName OR hm.latin_name = $map[h.scientificName]
        MATCH (hm)-[:HAS_COMPOUND]->(:Compound)-[:TARGETS]->(p:Protein)
        WITH p, count(DISTINCT hm) as functional_hits
        WHERE functional_hits > 1
        RETURN p.name as gene, p.fullName as full_name
        ORDER BY functional_hits DESC
        LIMIT 20
        """
        
        HERB_NAME_MAP = {
            "Curcuma longa": "Curcumae Longae Rhizoma",
            "Piper longum": "Piperis Longi Fructus", 
            "Zingiber officinale": "Zingiberis Rhizoma",
            "Phyllanthus emblica": "Phyllanthi Fructus",
            "Tinospora cordifolia": "Tinosporae Radix",
            "Glycyrrhiza glabra": "Glycyrrhizae Radix Et Rhizoma"
        }

        with self.db.driver.session() as s:
            t_res = s.run(q_top_shared, herbs=herb_ids, map=HERB_NAME_MAP)
            top_genes = [r["gene"] for r in t_res]
        
        if top_genes:
            explanation.append("\nKey Biological Mechanisms (Enriched):")
            categories = {}
            for gene in top_genes:
                meta = tk.enrich_target(gene)
                if meta:
                    cat = meta['category']
                    if cat not in categories: categories[cat] = []
                    categories[cat].append(f"**{gene}** ({meta['description']}) [Ref: {meta['citation']}]")
            
            for cat, lines in categories.items():
                explanation.append(f"\n{cat}:")
                for line in lines:
                    explanation.append(f"- {line}")

        return {
            "shared_effects": shared_effects,
            "interactions": interactions,
            "vae_analysis": vae_analysis,
            "explanation": "\n".join(explanation)
        }

    def close(self):
        self.db.close()
