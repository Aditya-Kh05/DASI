"""
Text Generator — Synthetic Scam & Legitimate Conversation Generator.

Generates realistic Hindi/English/code-mixed text transcripts for
digital arrest scam scenarios across all 6 phases, plus legitimate
conversations as negative samples.

Each generated conversation includes:
  - Full multi-turn transcript
  - Per-message phase labels
  - Per-message NLP sub-score ground truth
  - Language tag (en/hi/mixed)
"""

import random
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class GeneratedMessage:
    """A single message in a conversation."""
    role: str                    # "scammer" or "victim"
    text: str
    phase: int                   # 0=legitimate, 1-6=scam phases
    phase_name: str
    language: str                # "en", "hi", "mixed"
    nlp_labels: Dict[str, float]  # Ground truth sub-scores


@dataclass 
class GeneratedConversation:
    """A complete conversation with labels."""
    conversation_id: str
    is_scam: bool
    messages: List[GeneratedMessage]
    scenario_type: str           # e.g., "digital_arrest_cbi", "legitimate_bank_call"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "conversation_id": self.conversation_id,
            "is_scam": self.is_scam,
            "scenario_type": self.scenario_type,
            "num_messages": len(self.messages),
            "messages": [
                {
                    "role": m.role,
                    "text": m.text,
                    "phase": m.phase,
                    "phase_name": m.phase_name,
                    "language": m.language,
                    "nlp_labels": m.nlp_labels,
                }
                for m in self.messages
            ],
            "metadata": self.metadata,
        }


class TextGenerator:
    """Generates synthetic scam and legitimate conversation transcripts.
    
    The generator uses template-based generation with randomized entity
    injection (names, amounts, case numbers, legal sections) to create
    diverse but realistic transcripts.
    """
    
    # ════════════════════════════════════════════════════════
    # Scam Phase Templates
    # ════════════════════════════════════════════════════════
    
    # Phase 1: AUTHORITY_CLAIM — Scammer establishes credibility
    PHASE_1_TEMPLATES = {
        "en": [
            "Hello, this is {rank} {name} calling from {agency} headquarters, {city}. We have received a serious complaint registered against your {id_type} number.",
            "Good afternoon, I am {rank} {name} from the {agency}. Your {id_type} has been flagged in our system for suspicious activity. This is an urgent matter.",
            "Am I speaking with the holder of {id_type} ending {id_suffix}? This is {rank} {name}, {agency} {division}. We need to discuss an urgent legal matter concerning your identity.",
            "This call is from {agency}, {city} branch. I am {rank} {name} from the {division} division. Your identity documents have been found in connection with an ongoing investigation.",
            "Namaste, I am {rank} {name} from {agency}. We are conducting a high-priority investigation and your {id_type} number has come up in our records.",
        ],
        "hi": [
            "नमस्ते, मैं {rank} {name} बोल रहा हूं {agency} हेडक्वार्टर्स, {city} से। आपके {id_type} नंबर के खिलाफ एक गंभीर शिकायत दर्ज हुई है।",
            "मैं {rank} {name}, {agency} से बोल रहा हूं। आपकी {id_type} हमारे सिस्टम में संदिग्ध गतिविधि के लिए फ्लैग हुई है।",
            "क्या मैं {id_type} धारक से बात कर रहा हूं? यह {agency} की {division} शाखा से {rank} {name} बोल रहा है।",
        ],
        "mixed": [
            "Hello, main {rank} {name} bol raha hoon {agency} se. Aapke {id_type} number ke against ek serious complaint register hui hai.",
            "Namaste, this is {rank} {name} from {agency}, {city}. Aapki identity documents ek ongoing investigation mein aaye hain.",
            "Main {rank} {name}, {agency} {division} se call kar raha hoon. Aapke {id_type} ke saath kuch suspicious activity detect hui hai.",
            "Good morning, {agency} se {rank} {name} bol raha hoon. Aapka {id_type} number humari investigation mein flag hua hai.",
        ],
    }
    
    # Phase 2: ACCUSATION — Scammer makes formal accusations
    PHASE_2_TEMPLATES = {
        "en": [
            "Your {id_type} has been used to open {num_accounts} bank accounts that were involved in a money laundering case worth ₹{amount} crore. An FIR has been registered under sections {sections} of the {act}.",
            "We have evidence that your {id_type} was used in {crime_type}. The case number is {case_number}. Multiple transactions worth ₹{amount} crore have been traced to accounts linked to your identity.",
            "A warrant has been prepared in your name. The investigation reveals that {num_accounts} fraudulent accounts were opened using your {id_type}, and these accounts were used for {crime_type}.",
            "Our forensic analysis shows your {id_type} was compromised and used in an international {crime_type} network. The total amount involved is ₹{amount} crore under case {case_number}.",
        ],
        "hi": [
            "आपकी {id_type} का इस्तेमाल {num_accounts} बैंक अकाउंट खोलने में किया गया है जो ₹{amount} करोड़ की मनी लॉन्ड्रिंग में शामिल हैं। {act} की धारा {sections} के तहत FIR दर्ज है।",
            "हमारे पास सबूत हैं कि आपकी {id_type} का उपयोग {crime_type} में किया गया है। केस नंबर {case_number} है।",
            "आपके नाम पर वारंट तैयार किया गया है। जांच में पता चला है कि आपकी {id_type} से {num_accounts} फर्जी अकाउंट खोले गए।",
        ],
        "mixed": [
            "Aapki {id_type} se {num_accounts} bank accounts open kiye gaye hain jo ₹{amount} crore ki money laundering mein involved hain. FIR registered hai under {act} sections {sections}.",
            "Humare investigation mein aapki {id_type} {crime_type} case mein aayi hai. Case number hai {case_number}. Total ₹{amount} crore ka transaction trace hua hai.",
            "Aapke name pe warrant ready hai. {num_accounts} fraudulent accounts aapki {id_type} se linked hain, jo {crime_type} ke liye use kiye gaye.",
        ],
    }
    
    # Phase 3: FEAR_ESCALATION — Scammer escalates panic
    PHASE_3_TEMPLATES = {
        "en": [
            "A non-bailable arrest warrant has been issued in your name. If you do not cooperate, police will arrive at your residence within {hours} hours. This case also involves terrorism funding under {act}.",
            "The court has taken cognizance of this matter. Your passport will be suspended, all bank accounts will be frozen, and you could face {years} years of imprisonment.",
            "This is a very serious offense. Under {act}, the minimum punishment is {years} years. We have already informed the local police station in your area.",
            "I must inform you that {high_official} has personally reviewed this case. If you do not resolve this immediately, an arrest team will be dispatched to your location.",
            "Your name is being added to the Look Out Circular at all airports. All your financial assets will be seized. This is your last chance to cooperate before formal arrest proceedings begin.",
        ],
        "hi": [
            "आपके नाम पर गैर-जमानती गिरफ्तारी वारंट जारी हो चुका है। अगर आपने सहयोग नहीं किया तो {hours} घंटे में पुलिस आपके घर आ जाएगी।",
            "कोर्ट ने इस मामले का संज्ञान ले लिया है। आपका पासपोर्ट रद्द होगा, सभी बैंक अकाउंट फ्रीज होंगे, और {years} साल की सजा हो सकती है।",
            "यह बहुत गंभीर अपराध है। {act} के तहत कम से कम {years} साल की सजा है। हमने आपके क्षेत्र के थाने को पहले ही सूचित कर दिया है।",
        ],
        "mixed": [
            "Aapke naam pe non-bailable arrest warrant issue ho chuka hai. Agar aapne cooperate nahi kiya toh {hours} ghante mein police aapke ghar aa jayegi. This case involves {act} bhi.",
            "Court ne is case ka cognizance le liya hai. Aapka passport suspend hoga, saare bank accounts freeze honge, aur {years} saal ki jail ho sakti hai.",
            "Yeh bahut serious offense hai under {act}. Minimum {years} years ki punishment hai. Local police station ko already inform kar diya gaya hai.",
        ],
    }
    
    # Phase 4: ISOLATION — Scammer isolates victim
    PHASE_4_TEMPLATES = {
        "en": [
            "This is a classified investigation. You must NOT tell anyone — not your family, not your friends, not even your lawyer. Any disclosure is obstruction of justice under section {section} and carries additional charges.",
            "Stay on this video call. Do not disconnect. If you disconnect, it will be treated as non-cooperation and the arrest will proceed immediately.",
            "You are not allowed to discuss this case with anyone until the investigation is complete. This is a direct order from {high_official}. Any breach will result in immediate arrest.",
            "Do not contact your bank directly. We are monitoring all communications. Any attempt to alert others will be considered evidence tampering.",
            "Keep your camera on at all times. Our monitoring team is observing this session. Do not leave the frame or attempt to communicate with anyone else.",
        ],
        "hi": [
            "यह एक गोपनीय जांच है। आप किसी को नहीं बताएंगे — न परिवार को, न दोस्तों को, न वकील को। कोई भी खुलासा धारा {section} के तहत न्यायिक बाधा माना जाएगा।",
            "इस वीडियो कॉल पर बने रहिए। डिस्कनेक्ट मत कीजिए। अगर आपने कॉल काटा तो तुरंत गिरफ्तारी होगी।",
            "जांच पूरी होने तक आप इस केस के बारे में किसी से बात नहीं कर सकते। यह {high_official} का सीधा आदेश है।",
        ],
        "mixed": [
            "Yeh classified investigation hai. Aap kisi ko nahi batayenge — family ko nahi, friends ko nahi, lawyer ko bhi nahi. Any disclosure section {section} ke under obstruction of justice hai.",
            "Is video call pe bane rahiye. Disconnect mat kijiye. Agar aapne call disconnect kiya toh arrest immediately proceed hogi.",
            "Aap investigation complete hone tak kisi se is case ke baare mein baat nahi kar sakte. Yeh {high_official} ka direct order hai.",
        ],
    }
    
    # Phase 5: FINANCIAL_EXTRACTION — Scammer demands money
    PHASE_5_TEMPLATES = {
        "en": [
            "To clear your name and prove your innocence, you need to transfer all your funds to this RBI verification account {account_number} via NEFT immediately. This is a refundable security deposit.",
            "The court requires a security deposit of ₹{deposit_amount} lakh to process your bail application. Transfer to account number {account_number} in {bank_name}. This amount will be returned within {days} working days.",
            "For verification purposes, you must transfer your savings to the Supreme Court escrow account {account_number}. Once your identity is verified clean, the full amount will be refunded.",
            "The investigating officer needs you to make a compliance deposit of ₹{deposit_amount} lakh. This is standard procedure. Wire transfer to {account_number} at {bank_name}.",
            "To avoid arrest, transfer ₹{deposit_amount} lakh to the government verification account {account_number}. Share the transaction receipt immediately on this call.",
        ],
        "hi": [
            "अपनी बेगुनाही साबित करने के लिए आपको अपने सभी फंड RBI वेरिफिकेशन अकाउंट {account_number} में तुरंत ट्रांसफर करने होंगे। यह रिफंडेबल सिक्योरिटी डिपॉजिट है।",
            "कोर्ट को ₹{deposit_amount} लाख की सिक्योरिटी डिपॉजिट चाहिए। अकाउंट नंबर {account_number}, {bank_name} में ट्रांसफर करें।",
            "गिरफ्तारी से बचने के लिए ₹{deposit_amount} लाख सरकारी वेरिफिकेशन अकाउंट {account_number} में भेजें।",
        ],
        "mixed": [
            "Apni innocence prove karne ke liye aapko apne saare funds is RBI verification account {account_number} mein NEFT se transfer karne honge. Yeh refundable security deposit hai.",
            "Court ko ₹{deposit_amount} lakh ka security deposit chahiye bail application ke liye. Account {account_number}, {bank_name} mein transfer karein. {days} working days mein refund ho jayega.",
            "Arrest se bachne ke liye ₹{deposit_amount} lakh government verification account {account_number} mein transfer karein. Transaction receipt immediately share karein.",
        ],
    }
    
    # Phase 6: CONTINUED_CONTROL — Post-extraction continued manipulation
    PHASE_6_TEMPLATES = {
        "en": [
            "The verification is in process. You need to transfer an additional ₹{extra_amount} lakh for the supplementary investigation charges. The previous amount is being processed.",
            "The senior officer has found additional irregularities. Another deposit of ₹{extra_amount} lakh is required to complete the clearance process.",
            "Your case has been escalated to the high court. An additional security amount of ₹{extra_amount} lakh must be deposited to prevent immediate arrest.",
        ],
        "hi": [
            "वेरिफिकेशन प्रक्रिया चल रही है। सप्लीमेंटरी जांच के लिए ₹{extra_amount} लाख और ट्रांसफर करने होंगे।",
            "सीनियर अधिकारी को अतिरिक्त अनियमितताएं मिली हैं। क्लीयरेंस के लिए ₹{extra_amount} लाख और जमा करने होंगे।",
        ],
        "mixed": [
            "Verification process chal rahi hai. Supplementary investigation ke liye ₹{extra_amount} lakh aur transfer karne honge. Previous amount process ho raha hai.",
            "Senior officer ko additional irregularities mili hain. Clearance ke liye ₹{extra_amount} lakh aur deposit karna padega.",
        ],
    }
    
    # Victim responses for realism
    VICTIM_RESPONSES = {
        "confused": [
            "What? I don't understand. What complaint?",
            "Sir, there must be some mistake. I haven't done anything wrong.",
            "Kya? Mujhe samajh nahi aa raha. Kaisi complaint?",
            "But sir, I am a simple person. How can my Aadhaar be involved?",
            "Yeh kaise ho sakta hai? Maine kuch galat nahi kiya hai.",
        ],
        "scared": [
            "Oh my god, please sir. I am innocent. What should I do?",
            "Please don't arrest me. I will do whatever you say. My family will be devastated.",
            "Sir please mujhe bachao. Main innocent hoon. Kya karna padega?",
            "I'm begging you. Tell me what I need to do to clear my name.",
            "Bhagwan ke liye mujhe batao kya karna hai. Main bahut dar gaya hoon.",
        ],
        "compliant": [
            "Yes sir, I understand. I will not tell anyone. What do I need to do?",
            "Okay, I will transfer the money right away. Which account?",
            "Theek hai sir, main abhi transfer karta hoon. Account number batao.",
            "I am ready to cooperate fully. Just tell me the next steps.",
            "Haan sir, main samajh gaya. Main kisi ko nahi bataunga.",
        ],
        "questioning": [
            "But sir, how do I know this is real? Can I verify?",
            "Can I call the police station directly to confirm?",
            "Sir, mujhe thoda time dijiye verify karne ke liye.",
            "Is this really from the CBI? Can I see some identification?",
        ],
    }
    
    # ════════════════════════════════════════════════════════
    # Legitimate Conversation Templates
    # ════════════════════════════════════════════════════════
    
    LEGITIMATE_TEMPLATES = {
        "bank_call": [
            "Hello, this is {name} from {bank_name} customer service. We're calling regarding your recent credit card application. Your application reference number is {ref_number}.",
            "Good afternoon. This is a courtesy call from {bank_name}. We noticed you haven't activated your new debit card ending {card_suffix}. Would you like assistance?",
            "Hi, I'm calling from {bank_name} regarding your home loan pre-approval. We have some exciting offers for you based on your credit score.",
        ],
        "delivery_call": [
            "Hello, this is {name} from {courier}. Your package with tracking number {tracking} is out for delivery. Will someone be available at your address?",
            "Good morning, your order from {ecommerce} is arriving today. The delivery executive is {name}. Please share the OTP only with the executive at your door.",
        ],
        "telecom_call": [
            "Hello, this is {name} from {telecom}. We're calling to inform you about our new plan upgrade options. Your current plan expires next month.",
            "Good evening, I'm calling from {telecom} customer care regarding the complaint you raised about network issues in your area. Ticket number is {ticket}.",
        ],
        "insurance_call": [
            "Hello, I'm {name} from {insurance}. We're reaching out regarding the renewal of your health insurance policy number {policy}.",
            "Good afternoon. This is a reminder from {insurance} that your motor insurance is due for renewal in {days} days.",
        ],
        "social_call": [
            "Hey, how are you doing? Long time no talk! Are you free this weekend? We should catch up.",
            "Hi, this is {name} from the alumni association. We're organizing a reunion next month. Would you like to attend?",
            "Hello uncle, main {name} bol raha hoon. Mummy ne kaha aapko call karne ko. Sab theek hai na?",
        ],
    }
    
    # ════════════════════════════════════════════════════════
    # Entity Pools for Template Filling
    # ════════════════════════════════════════════════════════
    
    RANKS = ["Inspector", "Deputy Commissioner", "Senior Inspector", "ACP", "Sub-Inspector", "Superintendent"]
    NAMES_SCAM = ["Rajesh Sharma", "Vikram Singh", "Amit Kumar", "Suresh Patil", "Arjun Verma", "Deepak Malhotra", "Rakesh Mehta", "Sandeep Rao"]
    NAMES_LEGIT = ["Priya Sharma", "Ananya Gupta", "Ravi Kumar", "Sneha Patel", "Arjun Das", "Meera Reddy"]
    AGENCIES = ["CBI", "Enforcement Directorate", "Narcotics Control Bureau", "Cyber Crime Cell", "Mumbai Police Cyber Division", "Income Tax Department", "Customs Department"]
    CITIES = ["New Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Bangalore"]
    DIVISIONS = ["Cyber Crime", "Anti-Money Laundering", "Financial Investigation", "Narcotics", "Counter-Terrorism", "Economic Offenses Wing"]
    ID_TYPES = ["Aadhaar", "PAN card", "passport", "Aadhaar card"]
    CRIMES = ["money laundering", "drug trafficking", "terrorism funding", "tax evasion", "FEMA violation", "hawala transaction"]
    ACTS = ["PMLA Act", "UAPA", "Narcotics Act", "IT Act Section 66", "FEMA", "Prevention of Corruption Act"]
    SECTIONS = ["3, 4, and 70", "120B and 420", "153A and 505", "66C and 66D", "4 and 44"]
    HIGH_OFFICIALS = ["the Director General", "the Joint Commissioner", "the Chief Justice", "the Home Secretary", "the Additional Director General"]
    BANKS = ["State Bank of India", "HDFC Bank", "ICICI Bank", "Punjab National Bank", "Axis Bank", "Kotak Mahindra Bank"]
    COURIERS = ["BlueDart", "Delhivery", "DTDC", "FedEx India", "Amazon Logistics"]
    ECOMMERCE = ["Flipkart", "Amazon", "Myntra", "Meesho"]
    TELECOMS = ["Jio", "Airtel", "Vi (Vodafone Idea)", "BSNL"]
    INSURANCE = ["LIC", "HDFC Life", "Max Life", "Star Health", "ICICI Lombard"]
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._conversation_counter = 0
    
    def _fill_template(self, template: str) -> str:
        """Fill a template with random entities."""
        replacements = {
            "{rank}": self.rng.choice(self.RANKS),
            "{name}": self.rng.choice(self.NAMES_SCAM),
            "{agency}": self.rng.choice(self.AGENCIES),
            "{city}": self.rng.choice(self.CITIES),
            "{division}": self.rng.choice(self.DIVISIONS),
            "{id_type}": self.rng.choice(self.ID_TYPES),
            "{id_suffix}": str(self.rng.randint(1000, 9999)),
            "{crime_type}": self.rng.choice(self.CRIMES),
            "{act}": self.rng.choice(self.ACTS),
            "{sections}": self.rng.choice(self.SECTIONS),
            "{high_official}": self.rng.choice(self.HIGH_OFFICIALS),
            "{amount}": str(self.rng.choice([12, 18, 25, 32, 48, 56, 75, 100, 150])),
            "{num_accounts}": str(self.rng.randint(2, 8)),
            "{case_number}": f"RC-{self.rng.randint(100, 999)}/{self.rng.randint(2023, 2026)}",
            "{hours}": str(self.rng.choice([1, 2, 3, 4])),
            "{years}": str(self.rng.choice([5, 7, 10, 14, 20])),
            "{section}": str(self.rng.randint(100, 500)),
            "{account_number}": "".join(str(self.rng.randint(0, 9)) for _ in range(12)),
            "{bank_name}": self.rng.choice(self.BANKS),
            "{deposit_amount}": str(self.rng.choice([2, 3, 5, 10, 15, 25, 50])),
            "{extra_amount}": str(self.rng.choice([1, 2, 3, 5, 8])),
            "{days}": str(self.rng.choice([3, 5, 7, 10, 15])),
            # Legitimate call entities
            "{ref_number}": f"APP-{self.rng.randint(100000, 999999)}",
            "{card_suffix}": str(self.rng.randint(1000, 9999)),
            "{courier}": self.rng.choice(self.COURIERS),
            "{tracking}": f"BL{self.rng.randint(1000000, 9999999)}",
            "{ecommerce}": self.rng.choice(self.ECOMMERCE),
            "{telecom}": self.rng.choice(self.TELECOMS),
            "{ticket}": f"SR-{self.rng.randint(100000, 999999)}",
            "{insurance}": self.rng.choice(self.INSURANCE),
            "{policy}": f"POL-{self.rng.randint(10000, 99999)}",
        }
        
        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)
        return result
    
    def _get_nlp_labels(self, phase: int) -> Dict[str, float]:
        """Generate ground-truth NLP sub-scores for a given scam phase."""
        # Base scores by phase (authority, urgency, isolation, coercion, legal)
        phase_profiles = {
            0: {"authority": 0.05, "urgency": 0.05, "isolation": 0.02, "coercion": 0.03, "legal": 0.05},
            1: {"authority": 0.85, "urgency": 0.30, "isolation": 0.08, "coercion": 0.10, "legal": 0.70},
            2: {"authority": 0.75, "urgency": 0.65, "isolation": 0.15, "coercion": 0.45, "legal": 0.88},
            3: {"authority": 0.70, "urgency": 0.92, "isolation": 0.30, "coercion": 0.78, "legal": 0.85},
            4: {"authority": 0.60, "urgency": 0.80, "isolation": 0.95, "coercion": 0.85, "legal": 0.65},
            5: {"authority": 0.50, "urgency": 0.88, "isolation": 0.75, "coercion": 0.97, "legal": 0.55},
            6: {"authority": 0.45, "urgency": 0.85, "isolation": 0.70, "coercion": 0.95, "legal": 0.50},
        }
        
        profile = phase_profiles.get(phase, phase_profiles[0])
        # Add noise for realism
        noisy = {}
        for key, val in profile.items():
            noise = self.rng.gauss(0, 0.05)
            noisy[key] = max(0.0, min(1.0, val + noise))
        return noisy
    
    def _choose_language(self) -> str:
        """Randomly pick a language for a message."""
        return self.rng.choices(
            ["en", "hi", "mixed"],
            weights=[0.4, 0.25, 0.35],
            k=1,
        )[0]
    
    def generate_scam_conversation(
        self,
        num_phases: Optional[int] = None,
        force_language: Optional[str] = None,
    ) -> GeneratedConversation:
        """Generate a complete digital arrest scam conversation.
        
        Args:
            num_phases: How many phases to include (1-6). If None, random 4-6.
            force_language: Force all messages to this language. If None, random.
        
        Returns:
            GeneratedConversation with fully labeled messages.
        """
        self._conversation_counter += 1
        conv_id = f"scam_{self._conversation_counter:05d}"
        
        if num_phases is None:
            num_phases = self.rng.randint(4, 6)
        num_phases = min(6, max(1, num_phases))
        
        phase_templates = {
            1: self.PHASE_1_TEMPLATES,
            2: self.PHASE_2_TEMPLATES,
            3: self.PHASE_3_TEMPLATES,
            4: self.PHASE_4_TEMPLATES,
            5: self.PHASE_5_TEMPLATES,
            6: self.PHASE_6_TEMPLATES,
        }
        
        phase_names = {
            1: "AUTHORITY_CLAIM",
            2: "ACCUSATION",
            3: "FEAR_ESCALATION",
            4: "ISOLATION",
            5: "FINANCIAL_EXTRACTION",
            6: "CONTINUED_CONTROL",
        }
        
        messages = []
        
        for phase in range(1, num_phases + 1):
            lang = force_language or self._choose_language()
            templates = phase_templates[phase]
            
            # Pick template in chosen language
            if lang in templates and templates[lang]:
                template = self.rng.choice(templates[lang])
            else:
                # Fallback to English
                template = self.rng.choice(templates["en"])
                lang = "en"
            
            scammer_text = self._fill_template(template)
            scammer_msg = GeneratedMessage(
                role="scammer",
                text=scammer_text,
                phase=phase,
                phase_name=phase_names[phase],
                language=lang,
                nlp_labels=self._get_nlp_labels(phase),
            )
            messages.append(scammer_msg)
            
            # Add victim response (not always, and varied types)
            if self.rng.random() < 0.7:
                if phase <= 2:
                    response_type = self.rng.choice(["confused", "questioning"])
                elif phase <= 3:
                    response_type = self.rng.choice(["scared", "questioning"])
                else:
                    response_type = self.rng.choice(["scared", "compliant"])
                
                victim_text = self.rng.choice(self.VICTIM_RESPONSES[response_type])
                victim_msg = GeneratedMessage(
                    role="victim",
                    text=victim_text,
                    phase=phase,
                    phase_name=phase_names[phase],
                    language="mixed" if any(c > "\u0900" for c in victim_text) else "en",
                    nlp_labels=self._get_nlp_labels(0),  # Victim messages are not scam-scored
                )
                messages.append(victim_msg)
        
        # Add some extra scammer messages in critical phases for variation
        if num_phases >= 3 and self.rng.random() < 0.5:
            lang = force_language or self._choose_language()
            extra_phase = self.rng.choice([3, 4])
            templates = phase_templates[extra_phase]
            if lang in templates and templates[lang]:
                extra_template = self.rng.choice(templates[lang])
            else:
                extra_template = self.rng.choice(templates["en"])
                lang = "en"
            
            extra_msg = GeneratedMessage(
                role="scammer",
                text=self._fill_template(extra_template),
                phase=extra_phase,
                phase_name=phase_names[extra_phase],
                language=lang,
                nlp_labels=self._get_nlp_labels(extra_phase),
            )
            # Insert before financial extraction
            insert_idx = len(messages) - (2 if num_phases >= 5 else 0)
            messages.insert(max(0, insert_idx), extra_msg)
        
        scenario_type = self.rng.choice([
            "digital_arrest_cbi", "digital_arrest_ed", "digital_arrest_customs",
            "digital_arrest_narcotics", "digital_arrest_cybercrime",
        ])
        
        return GeneratedConversation(
            conversation_id=conv_id,
            is_scam=True,
            messages=messages,
            scenario_type=scenario_type,
            metadata={
                "num_phases": num_phases,
                "total_messages": len(messages),
            },
        )
    
    def generate_legitimate_conversation(self) -> GeneratedConversation:
        """Generate a legitimate (non-scam) conversation."""
        self._conversation_counter += 1
        conv_id = f"legit_{self._conversation_counter:05d}"
        
        scenario_type = self.rng.choice(list(self.LEGITIMATE_TEMPLATES.keys()))
        templates = self.LEGITIMATE_TEMPLATES[scenario_type]
        
        messages = []
        
        # 1-3 messages from the caller
        num_messages = self.rng.randint(1, 3)
        for _ in range(num_messages):
            template = self.rng.choice(templates)
            # Use legitimate names
            text = template.replace("{name}", self.rng.choice(self.NAMES_LEGIT))
            text = self._fill_template(text)
            
            msg = GeneratedMessage(
                role="caller",
                text=text,
                phase=0,
                phase_name="LEGITIMATE",
                language="en" if not any(c > "\u0900" for c in text) else "mixed",
                nlp_labels=self._get_nlp_labels(0),
            )
            messages.append(msg)
        
        return GeneratedConversation(
            conversation_id=conv_id,
            is_scam=False,
            messages=messages,
            scenario_type=scenario_type,
            metadata={"total_messages": len(messages)},
        )
    
    def generate_batch(
        self, 
        num_scam: int = 500,
        num_legitimate: int = 500,
    ) -> List[GeneratedConversation]:
        """Generate a balanced batch of scam and legitimate conversations."""
        conversations = []
        
        for _ in range(num_scam):
            conversations.append(self.generate_scam_conversation())
        
        for _ in range(num_legitimate):
            conversations.append(self.generate_legitimate_conversation())
        
        self.rng.shuffle(conversations)
        return conversations
