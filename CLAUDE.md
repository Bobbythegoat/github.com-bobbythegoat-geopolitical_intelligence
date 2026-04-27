Real-Time Geopolitical & Company Event Intelligence System
Goal

Build a decision-support system that:

Detects company-specific and geopolitical events as soon as they appear
Clusters duplicate reporting into single event objects
Scores credibility, relevance, and likely market impact
Produces readable cross-source summaries
Maps event transmission to sectors and tracked stocks
Sends alerts only when an event is credible and actionable
Maintains daily summaries and event history for feedback and model improvement
1. Product Definition
Core user problem

Public news is fragmented, repetitive, contradictory, and often late. The user needs one system that:

Tells them what happened
Separates confirmed facts from noise
Explains why it matters
Identifies which stocks and sectors may be affected
Indicates whether the narrative is early, crowded, or exhausted
Frames an actionable next step
Product promise

"When a material event breaks, the system ingests it, verifies it across sources, summarizes what matters, maps it to stocks, scores opportunity/risk, and alerts the user only when the signal clears quality thresholds."

2. MVP Scope
Narrow starting universe

Start with one high-signal sector and a small tracked universe.

Recommended starting sector

Semiconductors

Why semiconductors
Strong connection to geopolitics
Export controls and national security implications
Taiwan/supply chain exposure
AI infrastructure narrative
Foundry, equipment, memory, packaging, and hyperscaler links
Frequent news-driven repricing
Initial tracked stock universe (example)
NVDA
AMD
TSM
ASML
AMAT
LRCX
KLAC
MU
AVGO
ARM
INTC
QCOM
MRVL
SNPS
CDNS
TER
ON
NXPI
TXN
ADI
SMCI
MUFG? (remove if non-core)
GFS
UMC
MCHP

Keep initial universe to 20–30 names.

Initial event classes
Company-specific: filings, earnings, guidance changes, product launches, supply agreements, executive changes
Sector-specific: export controls, sanctions, subsidies, policy changes, major capex announcements
Macro/geopolitical: war escalation, trade restrictions, elections with policy risk, major industrial policy developments
3. Functional Modules
A. Ingestion Layer
Purpose

Continuously pull new information from multiple source classes.

Source classes
Primary official sources
SEC filings
Investor relations press releases
Government and regulator announcements
Central bank releases
Institutional-grade media
Reuters
Bloomberg
FT
WSJ
Specialist/trade sources
Semiconductor trade outlets
Policy/defense or energy-specific publications depending on chosen sector
Early-detection/noisy sources
X/Twitter lists
Reddit monitoring
newsletters
selected podcasts/transcript feeds
Ingestion methods
RSS polling
REST APIs
WebSocket streams
Scheduled scraping where terms permit
Email parsing for official company releases if needed
Output fields for each raw item
source_name
source_tier
source_url
publish_timestamp
headline
body_text
entities_mentioned
ticker_candidates
geography_tags
theme_tags
language
translated_text_if_needed
B. Preprocessing Layer
Goals
Clean raw text
Normalize timestamps
Translate if required
Extract entities and themes
Prepare records for clustering
Tasks
HTML stripping
Boilerplate removal
Named entity recognition
Ticker/entity resolution
Topic classification
Duplicate candidate generation
Language detection and translation normalization
C. Event Clustering Layer
Purpose

Turn many articles into one event object.

Why it matters

Without clustering, duplicate coverage creates false confidence.

Event object fields
event_id
canonical_title
first_seen_at
last_updated_at
event_type
involved_entities
affected_geographies
affected_tickers
source_count
primary_source_present (yes/no)
contradiction_flag
update_velocity
event_status
Clustering logic
headline/body semantic similarity
entity overlap
publication time proximity
topic match
contradiction detection
Event statuses
new
developing
confirmed
disputed
stale
resolved
D. Credibility Engine
Goal

Estimate how trustworthy and market-relevant the event is.

Inputs
source tier
primary source present
corroboration count
named vs anonymous sourcing
contradiction count
historical source reliability
official confirmation status
direct document evidence present
Sample scoring dimensions
Source quality score
Corroboration score
Primary-source proximity score
Contradiction penalty
Official confirmation bonus
Update stability score
Output categories
Confirmed
Likely credible
Unverified
Conflicted
Low-trust noise
Output fields
credibility_score (0–100)
confidence_label
evidence_summary
unresolved_questions
E. Relevance & Transmission Engine
Goal

Map the event to actual market impact.

Questions to answer
Which companies are directly affected?
Which sectors are second-order beneficiaries or losers?
What time horizon does the event matter on?
Is the effect operational, regulatory, demand-side, supply-side, or narrative-only?
Transmission categories
Direct fundamental impact
Supply chain impact
Regulatory/policy impact
Sentiment/narrative impact
Macro spillover impact
Time-horizon labels
Intraday
Short-term (1–5 trading days)
Medium-term (1–8 weeks)
Structural (multi-quarter)
Output fields
affected_stocks_ranked
affected_sectors_ranked
impact_direction
impact_horizon
first_order_effects
second_order_effects
F. Narrative Stage Engine
Goal

Determine whether the story is early or overcrowded.

Inputs
headline velocity
source breadth
social spread
analyst pickup rate
repeated phrase density
search trend proxies if available
price response vs attention response
Narrative stages
Early
Building
Consensus
Crowded
Euphoric
Exhausted
Why it matters

This is the core timing edge:

enter before broad crowding
avoid buying when the whole market already knows
identify when enthusiasm is peaking but marginal price response weakens
Output fields
narrative_stage
narrative_score
crowding_score
saturation_risk
G. Market Reaction Engine
Goal

Measure whether price has already absorbed the event.

Inputs
abnormal price move
abnormal volume
options activity changes
sector-relative move
gap vs historical volatility
time since event detection
Core outputs
price_reaction_lag_score
move_already_priced_flag
volatility_shift
divergence_between_attention_and_price
Key signal examples
Strong event + weak price move = possible underreaction
Huge attention + slowing price response = possible narrative exhaustion
H. Stock Scoring Engine
Goal

Convert event and narrative analysis into stock-specific scorecards.

Separate score components
Event Exposure Score
Credibility-Weighted Impact Score
Narrative Stage Score
Crowding Score
Price-Reaction Lag Score
Risk Score
Asymmetry Score
Overall Opportunity Score
Example scorecard schema
ticker
event_exposure_score
impact_score
credibility_score
narrative_stage_score
crowding_score
lag_score
risk_score
opportunity_score
confidence_score
suggested_decision_bucket
Decision buckets
Watch
Research deeper
Early opportunity
Wait for confirmation
Avoid due to crowding
Reduce exposure
Possible exit window approaching
I. Summary Engine
Goal

Create readable combined summaries across all sources.

Per-event summary structure
What happened
What is confirmed
What is disputed
What remains unknown
Why it matters
Affected sectors/stocks
Bull case / bear case / neutral case
Confidence level
Daily summary structure
Top 5–10 events of the day
What changed since previous summary
Which narratives strengthened
Which narratives weakened
Stocks with strongest score changes
Watchlist changes
Style requirements
concise but dense
plain English
no hype language
evidence-backed claims
explicit uncertainty markers
J. Alerting Engine
Goal

Notify only when an event is likely worth attention.

Alert trigger conditions
Event is new or materially updated
Credibility above threshold
Relevance above threshold
Expected market impact above threshold
Narrative not already fully saturated
Alert tiers
Critical
Official high-impact event
Very high credibility
Direct effect on tracked names
High priority
Strong corroboration
Significant sector implications
Monitor
Early but not fully confirmed
Ignore/no alert
weak source quality
low relevance
duplicate noise
Alert delivery channels
Telegram
Slack
SMS for top-tier alerts only
Email digest
Dashboard notification center
Alert card format
Event title
Timestamp
Credibility level
Impact level
Stage label
Key affected stocks
One-paragraph summary
Decision bucket
K. Dashboard / User Interface
Main panels
Live event feed
Top alerts
Daily combined brief
Stock scorecards
Narrative tracker
Watchlist status
Event history and feedback
Key views
Event detail view
consolidated summary
source list
evidence chain
contradictions
impacted entities
score timeline
Stock detail view
current scorecard
active event exposures
narrative stage history
recent alerts
decision bucket timeline
Heatmaps
sector heatmap by impact
watchlist heatmap by opportunity score
crowding heatmap
4. Technical Architecture
Suggested stack for MVP
Backend: Python (FastAPI)
Task orchestration: Celery / simple async workers initially
Messaging/queue: Redis queue first, Kafka later if needed
Database: PostgreSQL
Search/indexing: optional OpenSearch or pgvector later
Object storage: S3-compatible bucket for raw event snapshots
Frontend: Next.js or simple React dashboard
Alerting integrations: Telegram bot, Slack webhook, email service
LLM usage

Use LLMs only for bounded tasks:

summarization
contradiction extraction
event title generation
qualitative explanation
reasoned but constrained decision framing

Do not use LLMs as unconstrained forecasters.

Non-LLM components
entity extraction pipelines
rules engine
scoring formulas
clustering logic
thresholding and alert orchestration
5. Data Model
Core tables
sources
raw_articles
events
event_articles
tickers
event_ticker_impacts
stock_scores
alerts
user_watchlists
daily_briefs
model_feedback
Example table notes
events
event_id
canonical_title
first_seen_at
last_updated_at
event_status
event_type
credibility_score
narrative_stage
summary_text
stock_scores
ticker
timestamp
event_exposure_score
impact_score
crowding_score
lag_score
asymmetry_score
opportunity_score
decision_bucket
6. Scoring Framework
Example normalized formulas
Credibility score

Weighted average of:

source quality
corroboration density
primary evidence presence
contradiction penalty
official confirmation bonus
Opportunity score

Weighted average of:

event exposure
credibility-weighted impact
price reaction lag
asymmetry minus:
crowding penalty
risk penalty
Calibration process
start rule-based
manually review alerts
tune weights weekly
do not overfit on backtests early
7. Feedback Loop
Why this matters

If you do not evaluate event outcomes, the system stays decorative.

Review dimensions

For each alert, log:

Was the event genuinely material?
Was the credibility score accurate?
Did the market move in the expected direction?
Was the alert early, on time, or late?
Was the narrative stage classification accurate?
Would the decision bucket have been useful?
Weekly review
false positives
missed major events
score drift
best/worst alert examples
thresholds that need adjustment
8. Build Phases
Phase 1: Skeleton MVP
choose one sector and 20–30 stocks
connect 3–5 source streams
build raw ingestion pipeline
create event clustering
create manual credibility and relevance review workflow
send simple Telegram alerts
Phase 2: Scoring MVP
implement rule-based credibility engine
implement event-to-stock mapping
implement first scorecards
add daily brief generation
add dashboard v1
Phase 3: Narrative Engine
add source breadth and headline velocity metrics
add crowding and saturation logic
compare attention vs price response
improve decision buckets
Phase 4: Model Improvement
add more sources
expand to second sector
refine formulas using feedback logs
improve alert personalization
9. Operating Principles
Rules to prevent garbage output
Never collapse all logic into one score
Always show evidence and uncertainty
Use summaries to compress information, not replace reasoning
Prefer fewer high-quality alerts over many noisy ones
Keep a human override loop in all investment-facing outputs
Restrict recommendations to framed decision buckets rather than hard trade commands
10. First Deliverables
Immediate deliverables to build first
Tracked stock universe list
Source priority list
Event schema
Scoring schema
Telegram alert template
Daily brief template
Dashboard wireframe
Manual review rubric for first 100 alerts
11. Example Daily Workflow
Morning
Daily combined brief generated
Top active narratives updated
Stocks re-ranked by opportunity score
Intraday
New event detected
Cluster updated
Scores recalculated
Alert sent if thresholds exceeded
End of day
Final brief generated
Score changes logged
Alerts reviewed against actual price action
12. North Star Metrics
Product quality metrics
alert precision
alert recall on major market-moving events
average time from event publication to alert
percentage of alerts user marks useful
false positive rate
summary readability rating
Investment utility metrics
average post-alert abnormal return by decision bucket
percentage of early-opportunity alerts that preceded major narrative crowding
percentage of avoid-crowding alerts that correctly flagged late-stage narratives
13. Biggest Failure Modes
Too many alerts
Weak source weighting
Bad ticker/entity mapping
Duplicate events treated as separate signals
Overuse of LLMs without rule constraints
Fake precision in scoring
Monitoring too many stocks too early
No postmortem process
14. Final Positioning

This is not a generic stock-news bot.

It is: a real-time event intelligence system that detects company and geopolitical developments, verifies them across sources, summarizes what matters, maps impact to tracked stocks, scores timing and crowding, and alerts the user only when the signal is credible and worth attention.

15. Hybrid Product Form: Web Terminal First, Desktop Wrapper Later
Phase A: Browser-based terminal

Build the first version as a desktop-first web app.

Why this is the correct first move
fastest to build
easiest to debug
easiest to deploy
supports browser push notifications
can still feel like an app when pinned or installed as a PWA
Core features in browser phase
live dashboard
browser notifications
click-through event pages
watchlists
daily brief tab
event history and search
Phase B: Desktop wrapper

Once the browser version works, package it as a desktop app using Tauri.

Why Tauri later
native desktop notifications
clean app experience
lightweight compared with Electron
keeps same frontend/backend logic with minimal product rewrite
Notification behavior
Browser phase
browser push notifications
clicking notification opens event detail page or stock detail page
backup alerts through Telegram if browser is closed
Desktop phase
native OS notifications
notification click opens exact event card in app
pinned always-on terminal behavior possible
16. Dashboard Wireframe
A. Main Layout

Think of the interface as a compact intelligence terminal, not a generic finance dashboard.

Global page structure
Top navigation bar
Left sidebar
Main center analysis panel
Right intelligence panel
Visual priority
center panel = event understanding
right panel = stock consequences
left panel = navigation and live alert flow
B. Top Navigation Bar
Purpose

Persistent control layer for searching, filtering, and switching modes.

Sections
Left
product name / logo
market status indicator
last refresh timestamp
Center
universal search bar
search company
search ticker
search event
search theme
search country or region
Right
filter dropdowns
sector filter
event type filter
credibility filter
narrative stage filter
notification bell
settings/profile icon
Top bar behavior
always visible
supports keyboard shortcut focus for search
should show whether live stream is active
C. Left Sidebar
Purpose

Navigation + live situational awareness.

Sections
1. Main navigation
Dashboard
Live Feed
Watchlist
Daily Brief
Narratives
Alerts History
Settings
2. Watchlist quick-access

Pinned list of tracked names.

For each ticker show:

ticker symbol
current opportunity score color
active event count
trend arrow
3. Live alert stream

Compact stacked cards showing newest material events.

Each alert card should show:

small severity dot
event title
timestamp
affected tickers
credibility tag
Sidebar behavior
clicking alert loads event details in center panel
clicking ticker loads stock detail page
should support collapse/expand
D. Main Dashboard Home View
Purpose

High-level command center when you first open the terminal.

Top section: Active environment snapshot

Four compact summary cards:

Top geopolitical risk today
Top company event today
Most crowded narrative
Best early-stage opportunity cluster

Each card contains:

one-line title
short summary
confidence marker
click-through link
Middle section: Today’s top events table

Columns:

event title
type
credibility
impact
narrative stage
top affected stocks
decision bucket
time detected
Bottom section: Opportunity movers

Two ranked lists:

biggest score increases today
biggest score decreases today
E. Event Detail View
Purpose

This is the core page. When you click an alert, this page should make the event instantly understandable.

Structure
Header strip
event title
event type
timestamp first seen
last updated time
credibility badge
impact badge
narrative stage badge
decision bucket badge
Section 1: What happened

One clean paragraph in plain English.

Section 2: Consensus and uncertainty

Three columns or stacked blocks:

Confirmed
Disputed
Unknown
Section 3: Why it matters

Explain the transmission mechanism.

Subsections:

first-order effects
second-order effects
likely time horizon
what would invalidate the thesis
Section 4: Source evidence

List the source stack by trust level.

Show:

official source
top corroborating sources
lower-trust or early-detection sources
contradiction notes

Each source item should display:

source name
source tier
timestamp
short note on what it contributed
Section 5: Market impact map

Heatmap or ranked list of:

affected sectors
positively exposed stocks
negatively exposed stocks
uncertain exposure stocks
Section 6: Linked stocks scorecards

Each relevant stock gets a mini-card containing:

ticker
exposure score
opportunity score
crowding score
lag score
risk score
recommendation bucket
Section 7: Narrative development timeline

Timeline showing:

event first detected
official confirmation
major source pickup points
notable price reaction changes
F. Stock Detail View
Purpose

When you click a stock, you should see all active event exposure in one place.

Header area
ticker and company name
current opportunity score
active narrative stage
number of active event links
current watchlist status
Main sections
1. Current scorecard

Display all modular scores clearly:

event exposure
credibility-weighted impact
narrative stage
crowding
lag
risk
asymmetry
overall opportunity
2. Active event exposures

List all open events affecting the stock.

For each event show:

event title
direction of impact
credibility
stage
last updated
3. Summary of current thesis

Three blocks:

bull case
bear case
neutral/uncertain case
4. Recent score history

Small chart or table of:

opportunity score over time
crowding score over time
major event markers
5. Suggested decision framing

Constrained recommendation bucket only. Examples:

Watch
Research deeper
Early opportunity
Wait for confirmation
Avoid due to crowding
Possible exit window approaching
G. Daily Brief View
Purpose

Give you one readable summary of the entire day.

Sections
1. Executive summary

A short high-level narrative of the day.

2. Top global developments

Ranked summary cards.

3. Top company developments

Ranked summary cards.

4. Narrative tracker

Which themes are:

strengthening
weakening
becoming crowded
beginning to exhaust
5. Watchlist changes

Names that materially changed score or status.

6. Tomorrow setup

Potential follow-up events or risk areas to watch.

H. Alerts History View
Purpose

Archive and review all prior alerts.

Columns
timestamp
event title
alert tier
credibility
impact
decision bucket
user feedback tag
Required actions
mark useful / not useful
mark early / on time / late
add manual notes

This is critical for improving the system over time.

I. Narratives View
Purpose

Track major themes across time rather than isolated events.

Example narratives
AI infrastructure demand
China export restrictions
Taiwan geopolitical risk
defense spending escalation
oil supply shock
For each narrative show
current stage
velocity
breadth of coverage
key affected stocks
crowding level
confidence level

This page helps you avoid thinking in isolated headlines.

J. Notification Experience
Browser notification format

Title:

High Impact Event Detected

Body:

one-line event summary
top affected tickers
credibility label
stage label

Example:

Export control escalation may affect ASML, AMAT, LRCX
Credibility: High
Stage: Early-building
Notification click behavior
opens exact event detail page
if app already open, focuses current tab/window
if closed, launches browser terminal to event route
Notification rules
do not notify for low-credibility duplicates
suppress repeated alerts for same event unless materially updated
allow user-specific quiet hours if needed
K. Priority Views for Version 1

Do not build everything at once.

Must-have V1 views
Dashboard home
Event detail view
Stock detail view
Daily brief view
Live alert sidebar
Delay until later
advanced charts
portfolio integration
heavy customization
multi-user features
full mobile optimization