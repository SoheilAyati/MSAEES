# Wissenschaftliches Poster - Vorentwurf

**Arbeitstitel:**  
**Non-Intrusive Load Monitoring mit harmonischer Feature-Fusion und Live-Lernen**

**Projekt:** Modeling, Simulation and Automation of Electrical Energy Systems  
**Autoren:** Soheil Ayati, Marc Steffgen  
**Stand:** erster Posterentwurf auf Basis der Dokumente in `docs/`

---

## 0. Zentrale Poster-Story

Wir entwickeln ein NILM-System, das aus einer einzigen Messstelle am Point of Common Coupling erkennt, welche Verbraucher gerade aktiv sind und wie viel Leistung sie beitragen. Der Fokus liegt nicht nur auf einer Offline-Klassifikation, sondern auf einer durchgängigen Labor-Pipeline: synthetische Daten mit Ground Truth, reale Messungen mit einem Siemens PAC4200, physikalisch begründete Aggregation, Feature-Fusion aus Leistung, Blindleistung, Harmonischen und Schaltvorgängen sowie ein Live-System, das unbekannte Geräte nachträglich lernen kann.

**Kernbotschaft für das Poster:**  
Ein industrieller 5-Hz-Leistungsmesser kann für praxisnahes NILM nützlich sein, wenn seine verfügbaren Messgrößen gezielt kombiniert werden und das System unbekannte Lasten nicht erzwingt, sondern als Residual erkennt und nachlernt.

---

## 1. Vorgeschlagene Posteraufteilung in 4 Fenster

```text
+-----------------------------+-----------------------------+
| 1 Motivation & Fragestellung | 2 Methodik & Datenpipeline  |
| Warum NILM? Warum schwierig? | Synthetisch + real + HDF5   |
| PV, ähnliche Lasten, Meter   | PAC4200, Aggregation, ML    |
+-----------------------------+-----------------------------+
| 3 Umsetzung & Live-System    | 4 Ergebnisse & Ausblick     |
| Features, Modelle, Dashboard | Kennzahlen, Stand, Grenzen  |
| Training-on-the-go           | Verbesserungen              |
+-----------------------------+-----------------------------+
```

Optional oben quer: Titel, Autoren, Institution, ein kurzer Ein-Satz-Claim.  
Optional unten quer: QR-Code zum Repository / Paper, wichtigste Quellen, Kontakt.

---

## Fenster 1 - Motivation & Forschungsfrage

### Überschrift

**Warum Non-Intrusive Load Monitoring?**

### Kernaussage

Ein einzelner Zähler am Netzanschlusspunkt ist deutlich einfacher als Einzelmessungen an jedem Gerät. Die Herausforderung: Im Aggregatsignal überlagern sich mehrere Verbraucher, ähnliche Geräte haben ähnliche Wirkleistungen, und PV-Erzeugung kann Verbrauch teilweise verdecken oder das Vorzeichen der Gesamtleistung umkehren.

### Postertext

Non-Intrusive Load Monitoring (NILM) versucht, Geräteaktivität und Leistungsanteile aus einer einzigen aggregierten Messung zu rekonstruieren. Dadurch können Energiefeedback, Laborüberwachung und Lastanalyse ohne intrusive Unterzähler realisiert werden. In der Praxis ist dies schwierig, weil Geräte gleichzeitig laufen, ähnliche Wirkleistungen besitzen und PV-Erzeugung hinter dem Zähler das gemessene Signal maskieren kann.

**Forschungsfrage:**  
Wie kann ein NILM-System mit einem real verfügbaren industriellen Messgerät robuste Geräteerkennung ermöglichen, obwohl nur niederfrequente, verarbeitete Messgrößen statt Rohwellenformen verfügbar sind?

### Wichtige Punkte

- Eine Messstelle statt vieler Unterzähler.
- PV wird als signed power behandelt: Verbrauch positiv, Erzeugung negativ.
- Wirkleistung allein reicht nicht aus.
- Zusätzliche Signaturen: Blindleistung, Leistungsfaktor, THD, Harmonische, Schaltkanten.
- Ziel ist ein vollständiger Workflow von Messung bis Live-Anwendung.

### Visual-Idee

Ein einfaches Schema:

```text
Gerät A + Gerät B + PV + unbekannte Last
              |
              v
       PAC4200 am PCC
              |
              v
   NILM: Wer ist an? Wie viel W? Was bleibt unerklärt?
```

Mögliche Abbildung aus dem Projekt: Architekturdiagramm / MS1-Pipeline oder ein aggregiertes 24-h-Signal mit Ground Truth.

### Unterfolien / Detailfolien

1. **NILM-Prinzip:** Aggregatsignal vs. versteckte Gerätebeiträge.
2. **Problemfälle:** Überlagerung, ähnliche Wirkleistung, PV-Signal-Eclipse.
3. **Hardware-Realität:** PAC4200 liefert verarbeitete 5-Hz-Messwerte statt Rohwellenformen.

---

## Fenster 2 - Methodik & Datenpipeline

### Überschrift

**Eine gemeinsame Pipeline für synthetische und reale Daten**

### Kernaussage

Synthetische Daten liefern exakte Ground Truth, reale Einzelgeräte-Messungen liefern realistische Signaturen. Beide Datenquellen werden in ein gemeinsames HDF5-Format überführt, sodass Preprocessing, Feature-Extraktion, Training, Inferenz und Live-Betrieb denselben Codepfad nutzen.

### Postertext

Das Projekt kombiniert zwei Datenwelten. Synthetische Gerätegeneratoren erzeugen kontrollierbare Szenarien mit vollständiger Ground Truth pro Gerät und Zeitschritt. Reale PAC4200-Aufnahmen einzelner Geräte erfassen die tatsächlichen Labor-Signaturen. Ein gemessener Szenario-Mixer kombiniert reale Einzelgeräteaufnahmen mit zufälligen Ein-/Aus-Zeitplänen zu aggregierten Trainingsszenarien mit bekannter Ground Truth.

Die Aggregation ist physikalisch motiviert: Wirkleistung und Blindleistung werden phasenweise summiert, PV wird mit negativem Vorzeichen geführt, und Stromharmonische werden als komplexe Vektoren addiert, damit Verstärkung und Auslöschung zwischen gleichen Ordnungen realistisch abgebildet werden.

### Pipeline

```text
Synthetische Gerätegeneratoren       Reale PAC4200 Einzelmessungen
        |                                      |
        v                                      v
  synthetische Szenarien              gemessene Szenario-Mixe
        |                                      |
        +-------------- HDF5 -----------------+
                       |
                       v
       Preprocessing, Feature Engineering, Training
                       |
                       v
          Offline-Inferenz und Live-Dashboard
```

### Daten- und Feature-Konzept

- Gemeinsames Format: HDF5 mit Messwerten, Metadaten und Ground Truth.
- Messrate: 5 Hz, passend zur realistischen PAC4200-Abfrage über Modbus TCP.
- Feature-Fusion:
  - Steady-state: P, Q, S, Leistungsfaktor.
  - Harmonisch: THD, 3./5./7. Harmonische, Spektralenergie.
  - Phase: Leistung pro Phase.
  - Transient: maximale P/Q-Schaltstufe und Anzahl der Schritte.
- Fensterung: 10 s für gemessene Szenarien und Live-Betrieb.

### Visual-Idee

Ein Workflow-Diagramm mit zwei Eingängen: synthetisch und real.  
Mögliche Projektabbildungen: `docs/paper/figures/ms1_image1.png`, `docs/paper/figures/ms2_image2.png`.

### Unterfolien / Detailfolien

1. **Datenmodell:** Warum HDF5, Ground Truth und signed power?
2. **Aggregation:** Warum harmonische Vektorsumme statt Magnitudensumme?
3. **Feature Engineering:** Warum P/Q/PF/THD/Event-Features zusammengehören.
4. **Evaluation:** Gruppierte Splits, Makro-F1, gated MAE.

---

## Fenster 3 - Umsetzung & Live-System

### Überschrift

**Vom Modell zur laufenden Laboranwendung**

### Kernaussage

Das System erkennt nicht nur offline Geräte, sondern läuft live gegen den PAC4200 oder im Replay-Modus. Es zeigt aktive Geräte, Leistungsschätzungen, Confidence, Schaltkanten, erklärten Leistungsanteil und Residualleistung. Unbekannte Geräte können geführt aufgenommen und in das Modell nachtrainiert werden.

### Postertext

Die ML-Pipeline unterstützt vier Aufgaben: Identifikation einzelner Geräte, Disaggregation von Leistungsanteilen, Multi-Label-Präsenzschätzung und ein kombiniertes Mix-Modell für den Live-Betrieb. Als Standardmodell wird Random Forest verwendet, weil die Datensätze klein sind, die Features tabellarisch und interpretierbar sind und das Nachtrainieren schnell genug für den Live-Workflow bleiben muss. LightGBM und ein einfacher MLP dienen als Vergleichspfade.

Im Live-System werden Modellvorhersagen mit Schaltkantenerkennung kombiniert. Eine erkannte P/Q-Stufe kann ein Gerät direkt claimen, während das Fenster-Modell den stabilen Zustand bewertet. Residualleistung wird bewusst angezeigt: Wenn das System einen Leistungsanteil nicht erklären kann, meldet es ein unbekanntes Gerät statt die Leistung falsch auf bekannte Klassen zu verteilen.

### Live-Funktionen

- Verbindung zum Siemens PAC4200 über Modbus TCP.
- Sliding-window-Inferenz mit dem Mix-Modell.
- Event-Log mit Zeitstempeln für Schaltvorgänge.
- Presence-Smoothing und Hysterese gegen Flackern.
- Residual-basierte Unknown-Detection.
- Guided Teaching: Gerät isoliert aufnehmen, Szenarien neu mischen, Modelle nachtrainieren, Hot Reload.
- Replay-Modus für reproduzierbare Tests ohne angeschlossene Hardware.

### Visual-Idee

Screenshot des Live-Dashboards, idealerweise mit:

- erkannter Geräteliste,
- gestapelter Leistungsaufteilung,
- Residual / explained power,
- Event-Log.

Mögliche Projektabbildungen: `docs/Live_Test1.jpeg`, `docs/Live_Test2.jpeg`, `docs/Live_Test3.jpeg`.

### Unterfolien / Detailfolien

1. **Modellaufgaben:** identify, presence, disaggregate, mix.
2. **Live-Engine:** Ringbuffer, Fensterfeatures, Edge Detector, Dashboard.
3. **Unknown Handling:** Residualschwelle, Teach-Protokoll, Retraining.
4. **Warum Random Forest:** kleine Daten, Interpretierbarkeit, schnelles Nachlernen.

---

## Fenster 4 - Ergebnisse, aktueller Stand & Verbesserungen

### Überschrift

**Aktueller Stand: funktionierender Proof of Concept mit klaren nächsten Schritten**

### Kernaussage

Das Projekt erreicht eine vollständige End-to-End-Demonstration: Datenaufnahme, Training, Inferenz, Live-Erkennung, Residualanzeige und Nachlernen. Die bisherigen Ergebnisse sind vielversprechend, müssen aber durch mehr reale Messungen, echte PV-Erzeugung und stärkere Sequenzmodelle abgesichert werden.

### Postertext

Das aktuelle gemessene Mix-Modell wurde auf gemessenen Szenarien mit sieben Gerätefamilien trainiert. Auf gehaltenen Szenarien erreicht es eine Presence-Macro-F1 von 0,916 und eine gated power MAE von 2,9 W. Das Identify-Modell erreicht auf aktiven realen Gerätefenstern eine Macro-F1 von 0,955, allerdings mit einer optimistischen stratified-row Evaluation, weil pro Gerätefamilie bisher nur eine begrenzte Anzahl realer Sessions vorliegt.

Im Live-Test erkennt das System Schaltvorgänge, zeigt unerklärte Residualleistung und kann ein unbekanntes Gerät nach einer geführten Aufnahme nachtrainieren. In einem validierten Durchlauf wurde ein unbekanntes Gerät nach 8 s erkannt, das Modell in 26 s nachtrainiert und das Gerät anschließend mit 0,95 Confidence erkannt.

### Kennzahlen für das Poster

| Ergebnis | Aktueller Wert | Einordnung |
|---|---:|---|
| Mix Presence Macro-F1 | 0,916 | gemessene Szenarien, held-out |
| Gated Power MAE | 2,9 W | angezeigte, präsenz-gesteuerte Leistung |
| Identify Macro-F1 | 0,955 | reale Gerätefenster, stratified row split |
| Live Teach Loop | 26 s | Nachtrainieren + Hot Reload |
| Unknown Detection | nach 8 s | residualbasiert |
| Re-Recognition | 0,95 Confidence | nach Teaching |

### Grenzen

- Reale PV-Erzeugung ist noch nicht ausreichend validiert, da die verfügbare PV-Aufnahme nahezu 0 W erzeugt.
- Identify-Ergebnisse sind noch innerhalb weniger Sessions gemessen und brauchen gruppierte Evaluation über mehrere Aufnahmetage.
- Kleine Ventilatoren überlappen stark in P/Q und benötigen mehr Varianten.
- Harmonische Phasen sind synthetisch verfügbar, aber vom PAC4200 real nicht über Modbus messbar.
- Längere Gerätezyklen, z. B. Waschmaschine, benötigen Sequenzmodelle statt einzelner 10-s-Fenster.
- Gemessene Szenarien sind semi-synthetisch; echte simultane Mehrgeräteinteraktionen sollten stärker getestet werden.

### Verbesserungen / Ausblick

- Reale PV bei tatsächlicher Einspeisung aufnehmen.
- Mehr Sessions pro Gerät und mehrere Geräteinstanzen messen.
- Feature-Ablationen zeigen: common vs. harmonic vs. event features.
- CNN/LSTM oder Seq2Point für längere zeitliche Muster untersuchen.
- Schwierigkeitstiers sauber regenerieren: easy, normal, hard, adversarial.
- Live-Evaluation mit mehr unbekannten Geräten und echten Mehrgeräte-Szenarien erweitern.

### Visual-Idee

Links: Kennzahlen als kompakte Ergebnisbox.  
Rechts: Live-Screenshot oder Balkendiagramm der wichtigsten Scores.  
Unten: kurze Roadmap mit 3 nächsten Schritten.

### Unterfolien / Detailfolien

1. **Offline-Ergebnisse:** F1, MAE, Split-Logik.
2. **Live-Ergebnisse:** Unknown-Detection und Teach-Retrain-Recognise.
3. **Limitations:** PV, Datenmenge, Harmonische, Sequenzmodelle.
4. **Next Steps:** Messkampagne, Modellvergleich, Ablationen.

---

## Kompakte Posterfassung

### Ein-Satz-Claim

Wir zeigen eine vollständige NILM-Laborpipeline, die synthetische Ground-Truth-Daten, reale PAC4200-Messungen, harmonische Feature-Fusion und Live-Nachlernen kombiniert.

### Kurzabstract für das Poster

Non-Intrusive Load Monitoring rekonstruiert Geräteaktivität aus einer einzigen aggregierten Messung. In diesem Projekt wurde eine vollständige NILM-Pipeline für einen Laboraufbau mit Siemens PAC4200 entwickelt. Synthetische Szenarien liefern kontrollierbare Ground Truth, reale Einzelgeräteaufnahmen liefern Hardware-Signaturen, und ein gemeinsames HDF5-Format verbindet beide Quellen. Das Modell kombiniert Wirkleistung, Blindleistung, Leistungsfaktor, harmonische Merkmale und Schaltkanten. Im Live-System werden Präsenz, Leistung, Confidence und Residualleistung angezeigt; unbekannte Geräte können geführt aufgenommen und nachtrainiert werden. Das aktuelle Mix-Modell erreicht auf gemessenen Szenarien eine Presence-Macro-F1 von 0,916 und eine gated power MAE von 2,9 W. Die wichtigsten nächsten Schritte sind reale PV-Validierung, mehr Messsessions pro Gerät und Sequenzmodelle für längere Gerätezyklen.

---

## Designhinweise für das Poster

- **Leserichtung:** oben links Motivation, oben rechts Methode, unten links Umsetzung, unten rechts Ergebnisse.
- **Maximal 3-5 Bulletpoints pro Unterblock.**
- **Eine große Pipelinegrafik** ist wichtiger als viele kleine technische Details.
- **Kennzahlen sichtbar platzieren:** 0,916 F1, 2,9 W MAE, 26 s Retrain.
- **Grenzen offen nennen:** Das wirkt wissenschaftlicher und glaubwürdiger.
- **Live-Screenshots nutzen:** Sie zeigen, dass das Projekt über ein Notebook-Experiment hinausgeht.
- **Farbcode:** Blau für Messung/Pipeline, Grün für Ergebnisse, Orange/Rot für Residual/Unknown.

---

## Mögliche finale Posterüberschriften

1. **Non-Intrusive Load Monitoring mit harmonischer Feature-Fusion und Live-Lernen**
2. **Von einer Messstelle zur Geräteerkennung: NILM mit PAC4200 und Training-on-the-go**
3. **PV-aware NILM: Geräteerkennung aus aggregierten Leistungs- und Harmonischenmerkmalen**
4. **Ein Live-fähiges NILM-System für Laborlasten mit Residual-Erkennung**

