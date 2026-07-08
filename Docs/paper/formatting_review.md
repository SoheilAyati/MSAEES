# Formatting and Build Review

Erstellt am 2026-07-08 nach dem Build von `Docs/paper/main.tex` zu `main.pdf` mit 5 Seiten.

## Kritische Formatierungsprobleme

1. `Fig. 4` erscheint nach Beginn der References.
   - Sichtbar in der gerenderten PDF: Auf Seite 4 startet bereits der Abschnitt `References` mit Eintraegen [1] bis [5]. Auf Seite 5 erscheint danach noch `Fig. 4`, und erst darunter folgen die Referenzen [6] und [7].
   - Problem: Fuer ein Paper wirkt das ungeordnet, weil eine grosse Abbildung in den Literaturbereich hineinfloatet.
   - Moegliche Loesung: `Fig. 4` vor `\bibliographystyle`/`\bibliography` erzwingen, z. B. mit einer Float-Barriere vor den References oder durch andere Platzierung/Skalierung der Abbildung.

2. Seite 5 hat sehr viel ungenutzten Leerraum.
   - Ursache ist wahrscheinlich die spaete Platzierung von `Fig. 4` zusammen mit dem geteilten Literaturverzeichnis.
   - Problem: Die letzte Seite wirkt unfertig und unausgewogen.
   - Moegliche Loesung: `Fig. 4` frueher platzieren oder kleiner setzen, damit die References geschlossen am Ende stehen.

3. `Fig. 3` auf Seite 4 ist stark unausgewogen.
   - Die linke Subfigure sitzt deutlich tiefer als die rechte; links oben bleibt grosser Leerraum.
   - Problem: Die Abbildung wirkt optisch schief, obwohl sie technisch korrekt kompiliert.
   - Moegliche Loesung: Subfigures oben ausrichten, beide Bilder auf vergleichbare Hoehe bringen oder `ms2_image2.png`/`ms2_image3.png` vorab auf aehnliche Seitenverhaeltnisse zuschneiden.

4. `Fig. 2` auf Seite 3 nutzt den Raum nicht ideal.
   - Die zweigeteilte Abbildung steht sehr hoch mit viel leerem Raum ueber den Plots.
   - Die Subcaptions liegen eng nebeneinander; besonders der Uebergang zwischen `(a)` und `(b)` wirkt gequetscht.
   - Moegliche Loesung: Bilder einheitlich zuschneiden, etwas kleiner setzen, oder die Subcaptions kuerzen.

## Mittlere Formatierungsprobleme

5. Mehrere grosse `figure*`-Floats beeinflussen die Textreihenfolge stark.
   - Betroffen sind `Fig. 2`, `Fig. 3` und `Fig. 4`.
   - Problem: IEEE-Zweispaltensatz verschiebt breite Abbildungen oft auf spaetere Seiten. Dadurch stehen Discussion, Conclusion, References und Figuren nicht mehr in einer idealen Reihenfolge.
   - Moegliche Loesung: Nur die wichtigsten Abbildungen als `figure*` setzen; andere als einspaltige `figure` oder kleiner platzieren.

6. `Fig. 1` auf Seite 2 ist lesbar, aber relativ klein.
   - Die Architekturdetails sind sichtbar, koennten beim Druck aber knapp werden.
   - Moegliche Loesung: Grafik als breite `figure*` oder mit weniger Detail/mehr Kontrast exportieren.

7. `Table I` ist relativ dicht gesetzt.
   - Der Inhalt passt, aber einige Zellen sind textlastig und erzeugen Zeilenumbrueche.
   - Moegliche Loesung: Beschreibungen in der Tabelle kuerzen und Details in den Fliesstext verlagern.

## LaTeX-Warnungen aus dem Build

8. `Unused global option(s): [lettersize]`
   - Quelle: `\documentclass[lettersize,journal]{IEEEtran}`
   - Problem: `lettersize` wird von der Klasse nicht genutzt.
   - Moegliche Loesung: Wahrscheinlich `letterpaper` statt `lettersize` verwenden oder die Option entfernen.

9. `Underfull \hbox` in Zeilen 95--96.
   - Ursache: Zeilenumbruch im Abschnitt zum PAC4200 Reader.
   - Problem: Kein Build-Fehler, aber LaTeX meldet eine optisch lockere Zeile.
   - Moegliche Loesung: Satz leicht umformulieren oder manuell sinnvolle Trennung ermoeglichen.

10. `Underfull \hbox` in Zeilen 127--128.
    - Ursache: Text in `Table I`, vor allem lange Beschreibungen in schmalen Spalten.
    - Problem: Tabellenzeile wirkt typografisch etwas unruhig.
    - Moegliche Loesung: Tabelleninhalt kuerzen oder Spaltenbreiten anpassen.

11. `Underfull \hbox` in `main.bbl` bei Referenz [5].
    - Ursache: Langer Siemens-Titel/URL im Literaturverzeichnis.
    - Problem: Kein Fehler, aber unschoener Zeilenumbruch in der Referenz.
    - Moegliche Loesung: URL kuerzen, `url`-Breaks verbessern oder bibliografischen Eintrag kompakter formulieren.

12. MiKTeX meldet: `Bis jetzt haben Sie noch nicht nach MiKTeX-Updates gesucht.`
    - Das ist kein Dokumentproblem, sondern eine lokale MiKTeX-Hinweismeldung.
    - Moegliche Loesung: MiKTeX Console oeffnen und einmal nach Updates suchen.

## Inhaltliche/Strukturelle Beobachtungen

13. Die Abbildungen aus Milestone 1 bis 3 erzaehlen den Projektverlauf gut, nehmen aber viel Platz ein.
    - Bei nur 5 Seiten dominieren die breiten Screenshots und Plots stark.
    - Moegliche Loesung: Entweder weniger Abbildungen verwenden oder einige als zusammengesetzte, sauber zugeschnittene Summary-Figure exportieren.

14. Die References werden durch Float-Platzierung visuell zerteilt.
    - Das ist die wichtigste Sache vor einer Abgabe, weil es sofort auffaellt.
    - Prioritaet: zuerst Float-Reihenfolge fixen, danach Detailwarnungen.

## Empfohlene Reihenfolge fuer Korrekturen

1. `Fig. 4` vor die References bringen.
2. `Fig. 3` Subfigures optisch ausrichten oder Bilder zuschneiden.
3. `Fig. 2` Subcaptions/Skalierung verbessern.
4. `lettersize` durch `letterpaper` ersetzen oder entfernen.
5. Tabelle und lange Referenzzeilen bei Bedarf typografisch glaetten.
