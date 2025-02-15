import torch
import torch.nn as nn
import fitz  # PyMuPDF for PDF extraction
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class EndToEndResumeModel(nn.Module):
    def __init__(self, num_entity_labels, hidden_size=768):
        super(EndToEndResumeModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.entity_classifier = nn.Linear(hidden_size, num_entity_labels)
        self.match_classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask, job_input_ids=None, job_attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        entity_logits = self.entity_classifier(sequence_output)
        
        match_score = None
        if job_input_ids is not None:
            job_outputs = self.bert(job_input_ids, attention_mask=job_attention_mask)
            resume_cls = outputs.last_hidden_state[:, 0, :]
            job_cls = job_outputs.last_hidden_state[:, 0, :]
            combined = torch.cat([resume_cls, job_cls], dim=1)
            match_score = self.match_classifier(combined)
            # Apply a sigmoid to normalize the match score between 0 and 1
            match_score = torch.sigmoid(match_score)
        
        return entity_logits, match_score

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def get_resume_match_score(cv_text, job_text, model, tokenizer):
    resume_inputs = tokenizer(cv_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    job_inputs = tokenizer(job_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    model.eval()
    with torch.no_grad():
        _, match_score = model(resume_inputs['input_ids'], resume_inputs['attention_mask'],
                               job_input_ids=job_inputs['input_ids'], job_attention_mask=job_inputs['attention_mask'])

    return match_score.item()

def main():
    num_entity_labels = 10
    model = EndToEndResumeModel(num_entity_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Charger le CV depuis un fichier PDF
    cv_path = "cv2.pdf"  # Remplace par le chemin réel du CV
    cv_text = extract_text_from_pdf(cv_path)

    # Description de l'emploi
    job_text = (
        "We are looking for a software engineer with strong Python skills and experience in machine learning. "
        "Candidates with experience at major tech companies are preferred."
    )

    # Calculer le score d'adéquation
    match_score = get_resume_match_score(cv_text, job_text, model, tokenizer)
    
    # Convertir le score en pourcentage (de 0 à 100)
    normalized_score = max(0, min(100, match_score * 100))

    # Générer une conclusion
    if normalized_score > 80:
        conclusion = "Le candidat correspond parfaitement au poste."
    elif normalized_score > 60:
        conclusion = "Le candidat correspond bien au poste, avec quelques différences mineures."
    elif normalized_score > 40:
        conclusion = "Le candidat présente un certain potentiel, mais plusieurs écarts existent."
    else:
        conclusion = "Le candidat ne correspond pas suffisamment aux critères du poste."

    print("\n=== Résultat d'Analyse du CV ===")
    print(f"Score d'adéquation : {normalized_score:.2f}/100")
    print(f"Conclusion : {conclusion}")

if __name__ == '__main__':
    main()
