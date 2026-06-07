README / Begleitdokumentation zu den PAC4200-Messdaten
=========================================================

Projekt / Zweck
---------------
Die CSV-Dateien enthalten Messdaten eines Siemens SENTRON PAC4200, das über Modbus TCP ausgelesen wurde. Ziel ist die Erfassung von Geräte-Fingerprints unterschiedlicher Verbraucher. Dabei werden nicht nur Wirkleistung in Watt, sondern zusätzlich Scheinleistung, Blindleistung, Leistungsfaktor und Stromverzerrung erfasst.

Die Daten eignen sich zur Demonstration, dass elektrische Geräte nicht nur über ihren Verbrauch, sondern auch über ihr Netzverhalten charakterisiert werden können. Besonders relevant sind Einschaltvorgänge, stationärer Betrieb, Leistungsfaktor und THD des Stroms.

Messaufbau
----------
Messgerät:
Siemens SENTRON PAC4200 im Labor-/Demogehäuse

Kommunikation:
Modbus TCP über Ethernet

Logging:
Node-RED auf macOS

Abtastung:
nominal 200 ms

Hinweis zur Abtastung:
Die Messung erfolgt nominal mit 200 ms. Die tatsächlichen Messzeitpunkte werden über timestamp_iso in jeder Zeile gespeichert. Dadurch können kleinere Timing-Abweichungen des Loggers bei der späteren Auswertung berücksichtigt werden.

Messstrategie pro Gerät
-----------------------
Für jedes Gerät wird eine eigene CSV-Datei erzeugt. Der empfohlene Messablauf ist:

- 20 bis 30 s ohne angeschlossenes/eingeschaltetes Gerät als Baseline
- Gerät einstecken oder einschalten
- stationärer Betrieb über mindestens 40 bis 120 s
- Gerät ausschalten oder ausstecken
- 20 bis 30 s Nachlauf/Baseline

Warum Baseline?
Die Baseline zeigt, ob im Messaufbau eine Grundlast vorhanden ist. Bei sauberem Aufbau liegen Strom, Wirk-, Schein- und Blindleistung im ausgeschalteten Zustand nahe 0. Falls trotzdem kleine Werte auftreten, müssen diese bei der Interpretation berücksichtigt werden.

Beschreibung der CSV-Spalten
-----------------------------

timestamp_iso
Zeitstempel der Messzeile im ISO-Format. Der Zeitstempel ist die wichtigste Referenz für Ereignisse wie Einstecken, Einschalten, Lastwechsel oder Ausschalten.

device_name
Freitextbezeichnung des gemessenen Geräts, z. B. "LED Lampe 6.3 W" oder "USB Ladegerät + iPhone".

run_id
Eindeutige Kennung des Messlaufs. Sie sollte pro Messung angepasst werden, z. B. "led_6p3w_001" oder "usb_ladegeraet_iphone_001".

sample_interval_ms
Nominales Abtastintervall des Loggers. In den aktuellen Messungen ist der Zielwert 200 ms.

u_l1_n_v
Effektivwert der Spannung zwischen L1 und Neutralleiter in Volt. Dieser Wert dient zur Kontrolle der Netzspannung und zur Plausibilisierung strom- und leistungsabhängiger Werte.

i_l1_a
Effektivwert des Stroms auf L1 in Ampere. Dieser Wert ist für Geräte-Fingerprints wichtig, weil Geräte mit ähnlicher Wirkleistung unterschiedliche Effektivströme haben können.

p_total_w
Gesamtwirkleistung in Watt. Das ist der tatsächlich umgesetzte Leistungsanteil und der wichtigste Wert für den Energieverbrauch. Einschalt- und Ausschaltvorgänge sind in p_total_w meist gut erkennbar.

s_total_va
Gesamtscheinleistung in Voltampere, direkt aus dem PAC4200 gelesen. Sie beschreibt die elektrische Belastung des Netzes durch den Effektivstrom.

s_calc_va
Berechnete Scheinleistung aus u_l1_n_v * i_l1_a. Dieser Wert dient als Plausibilitätskontrolle zu s_total_va. Bei einphasiger Messung sollten s_total_va und s_calc_va nahe beieinander liegen.

q_total_var
Gesamtblindleistung in var. Sie beschreibt den Leistungsanteil, der zwischen Quelle und Verbraucher pendelt. Negative Werte deuten je nach Vorzeichenkonvention auf kapazitives Verhalten, positive Werte auf induktives Verhalten hin. Für Geräte-Fingerprints ist q_total_var wichtig, weil Motoren, Transformatoren, Schaltnetzteile und LED-Treiber unterschiedliche Blindleistungsanteile zeigen können.

pf_total
Gesamtleistungsfaktor. Er beschreibt das Verhältnis von Wirkleistung zu Scheinleistung. Eine ohmsche Last hat typischerweise einen Leistungsfaktor nahe 1. Elektronische Verbraucher wie LED-Lampen oder Ladegeräte können deutlich niedrigere Werte zeigen.

frequency_hz
Netzfrequenz in Hertz. Dieser Wert dient vor allem als Plausibilitäts- und Netzqualitätswert.

thd_u_l1_percent
Total Harmonic Distortion der Spannung auf L1 in Prozent. Der Wert beschreibt die Verzerrung der Netzspannung. Er ist eher Kontextinformation, kann aber helfen, Messbedingungen zu beurteilen.

thd_i_l1_percent
Total Harmonic Distortion des Stroms auf L1 in Prozent. Dieser Wert ist für Geräte-Fingerprints sehr wertvoll. Nichtlineare Verbraucher wie LED-Lampen, Ladegeräte und Schaltnetzteile können stark verzerrte Stromverläufe erzeugen und deshalb hohe THD-I-Werte zeigen.

block_time_difference_ms
Zeitdifferenz zwischen den beiden Modbus-Registerblöcken, aus denen eine gemeinsame Messzeile zusammengesetzt wurde. Kleine Werte bedeuten, dass Grundwerte und THD-Werte nahezu gleichzeitig gelesen wurden. Werte unter ca. 100 ms sind für diese Messung sehr gut. Größere Werte sollten bei schnellen Lastwechseln berücksichtigt werden.

Hinweise zur Interpretation
---------------------------

NaN-Werte bei pf_total oder thd_i_l1_percent
Wenn kein messbarer Strom fließt, können Leistungsfaktor und Strom-THD physikalisch nicht sinnvoll bestimmt werden. NaN-Werte in ausgeschalteten Phasen sind daher kein Fehler, sondern erwartbar.

Einsteckereignisse
Beim Einstecken eines Geräts können kurze transiente Ereignisse auftreten. Diese können sich als kurze Peaks in Strom, Leistung oder THD zeigen. Bei 200-ms-Abtastung werden solche Ereignisse nur grob erfasst. Das PAC4200 ist kein Oszilloskop und liefert keine Rohwellenform, sondern berechnete Messwerte.

Stationärer Betrieb
Für den Geräte-Fingerprint sollten Mittelwerte und Streuungen im stationären Bereich betrachtet werden, also nicht direkt während des Einsteckens oder Umschaltens.

Baseline
Die Aus-Phasen vor und nach der Messung sind wichtig, um sicherzustellen, dass keine relevante Grundlast, Status-LED oder Nebenverbraucher im Messpfad liegen.

Beispielinterpretation: LED-Lampe 6,3 W
---------------------------------------
Die LED-Lampe zeigte ungefähr:

- Wirkleistung: ca. 6,7 W
- Scheinleistung: ca. 9,0 VA
- Blindleistung: ca. -4,6 var
- Leistungsfaktor: ca. 0,74
- THD-I: ca. 48 %

Interpretation:
Die Lampe nimmt nur rund 6,7 W Wirkleistung auf, belastet das Netz aber mit rund 9 VA Scheinleistung und zieht einen deutlich verzerrten Strom. Das unterscheidet sie klar von einer ohmschen Last.

-----

Hinweise zu den Lasten:
LED Lampe: 20s aus -> 40s an -> 20s aus
USB Netzteil: 20s nicht eingesteckt -> 20s eingesteckt -> 40s eingesteckt mit iphone angeschlossen -> 20s nicht eingesteckt
Fön: 20s aus -> 40s Stufe 1 -> 20s aus
Mixer: 20s aus -> 20s Stufe 1 -> 20s Stufe 2 -> 20s aus
Toaster: 20s aus -> 40s an -> 20s aus (Achtung: Last größer 5A!)
Leuchtstoffröhre: 20s aus -> 40s an -> 20s aus