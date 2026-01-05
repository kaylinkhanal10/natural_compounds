class IntentParser:
    def parse(self, text: str):
        """
        Mock implementation of intent parsing.
        In a real scenario, this would call an LLM to extract biomedical entities.
        """
        text = text.lower()
        response = {
            "mapped_effects": [],
            "candidate_herbs": [],
            "rationale": "Intent mapping based on keyword matching (Demo Mode)."
        }
        
        if "immunity" in text or "immune" in text:
            response["mapped_effects"].append("EFF_IMMUNITY")
            response["candidate_herbs"].extend(["HERB_GUDUCHI", "HERB_AMLA", "HERB_TULSI"])
        
        if "digest" in text or "stomach" in text:
            response["mapped_effects"].append("EFF_DIGESTION")
            response["candidate_herbs"].extend(["HERB_GINGER", "HERB_PIPPALI", "HERB_LICORICE"])
            
        if "stress" in text or "anxiety" in text:
            response["mapped_effects"].append("EFF_ADAPTOGEN")
            response["candidate_herbs"].extend(["HERB_ASHWAGANDHA", "HERB_BRAHMI", "HERB_TULSI"])

        if "inflammation" in text or "pain" in text:
            response["mapped_effects"].append("EFF_ANTI_INFLAM")
            response["candidate_herbs"].extend(["HERB_TURMERIC", "HERB_GINGER"])
            
        return response
