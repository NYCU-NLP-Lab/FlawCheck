def get_aspect_prompt(claim, evidence):
    return f"""According to the provided evidence, enumerate four distinct aspects that can be employed to verify the authenticity of the claim.
Claim: {str(claim)[:200]}
Evidence: {str(evidence)[:1500]}
Consistently generate a numbered list of four distinct aspects."""


def get_flaw_prompt(claim, evidence, aspect):
    return f"""Based on the supplied evidence, scrutinize the specified claim to identify any potential flaws that may exist, taking into account the identified aspects.
Selected flaws: Contradicting facts, Exaggeration, Understatement, Occasional faltering, Insufficient support, Problematic assumptions, Existence of alternative explanations
Claim: {str(claim)[:200]}
Identified Aspects: {str(aspect)[:500]}
Evidence: {str(evidence)[:1500]}
Consistently generate a numbered list of results for selected flaws."""


def get_verdict_prompt(claim, flaw, aspect):
    return f"""Evaluate the validity of a given claim by examining identified aspects and potential flaws. Provide a comprehensive analysis.
Claim: {str(claim)[:200]}
Identified Aspects: {str(aspect)[:500]}
Potential flaws: {str(flaw)[:1500]}"""


def get_aspect_baseline_prompt(claim, evidence, aspect):
    return f"""Evaluate the authenticity of the claim by examining the provided evidence from four given unique aspects. Deliver a comprehensive analysis.
Claim: {str(claim)[:200]}
Identified Aspects: {str(aspect)[:500]}
Evidence: {str(evidence)[:1500]}"""


def get_baseline_prompt(claim, evidence):
    return f"""Validate the veracity of the given claim based on the provided evidence and generate a detailed explanation.
Claim: {str(claim)[:200]}
Evidence: {str(evidence)[:1500]}"""
