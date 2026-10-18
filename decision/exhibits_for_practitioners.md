# Practitioner exhibit cases (anonymized public-dataset examples)

Show during T8 sessions. Do not invent examples. Gold may be revealed after their judgment.

## W1-001 · TF_Reason · FP16

- **Why show:** see session guide
- **Input (truncated):**
  ```
  Use case name: AdvancedSearch 
  Description: The tourist searching for a site using the potential offered by the Advanced Search. 
  Participating Actor: initialized by Tourist 
  Entry conditions: The Tourist has successfully authenticated to the system. 
  Flow of events User System: 
  1. Enable the advanced search feature from your personal area. 
  2 View the advanced search form. 
  3 Fill in the form of advanced search and submit. 
  4 Gets the position of relying on the tourist event of the use location and process the request. 
  Exit conditions: The system displays a list of results. 
  Interruption of the connection to the server ETOUR. 
  Quality requirements:
  The system requirements into the transaction in more than 15 seconds.
  ```
- **Model output (raw):**
  ```
  ["AdvancedSearch.java", "Search.java", "StandardSearch.java"] 
  
  ```
- **Gold:** `['ISearch.java']`
- **Ask first (before gold):** Would your normal ~60s review catch/reject this?

## L2-019 (legacy Wave~2 ID W2-019) · TF_Surface · Q8_0

- **Why show:** see session guide
- **Input (truncated):**
  ```
  The product shall have an intuitive user interface.
  ```
- **Model output (raw):**
  ```
   US
  ```
- **Gold:** `F`
- **Ask first (before gold):** Would your normal ~60s review catch/reject this?

## L2-016 (legacy Wave~2 ID W2-016) · TF_Reason · Q8_0

- **Why show:** see session guide
- **Input (truncated):**
  ```
  Use case name: InsertCulturalHeritage 
  Description: Inserts a new cultural object in the system. 
  Participating Actor: initialization Agency Operator 
  Entry Operator conditions: The agency has logged. 
  Flow of events User System: 
  1. Activate the feature for the insertion of a new cultural good. 
  2 Displays the corresponding form. 
  3 Fill out the form with the data of the cultural and submit. 
  4 Verify the data entered and asks for confirmation of the transaction. Where the data is invalid or insufficient, the system activates the use case Errored. 
  5 Confirm the operation. 
  6. Memorize your new cultural good. 
  Exit conditions: The system shall notify the proper inclusion of the cultural. 
  Operator Agency cancels the operation. 
  Interruption of the connection to the server ETOUR. 
  Quality requirements:
  The system requirements will not accept duplicates cultural heritage.
  ```
- **Model output (raw):**
  ```
  
  - `BeanCulturalHeritage.java` implements the `InsertCulturalHeritage` use case. 
  - `DBCulturalHeritage.java` implements the `InsertCulturalHeritage` use case. 
  - `CulturalHeritage.java` implements the `InsertCulturalHeritage` use case. 
  - `CulturalHeritageAgencyManager.java` implements the `InsertCulturalHeritage` use case. 
  - `CulturalHeritageChecker.java` implements the `InsertCulturalHeritage` use case. 
  - `CulturalHeritageCommonManager.java` implements the `InsertCulturalHeritage` use case. 
  - `CulturalHeritageVisitChecker.java` implements the `InsertCulturalHeritage` use case. 
  
  
  
  ```json
  ["BeanCulturalHeritage.java", "DBCulturalHeritage.java", "CulturalHeritage.java", "CulturalHeritag
  …
  ```
- **Gold:** `['BeanCulturalHeritage.java', 'CulturalHeritage.java', 'CulturalHeritageAgencyManager.java', 'CulturalHeritageChecker.java', 'DBCulturalHeritage.java', 'ICulturalHeritageAgencyManager.java']`
- **Ask first (before gold):** Would your normal ~60s review catch/reject this?

## L2-005 (legacy Wave~2 ID W2-005) · TF_Reason · Q8_0

- **Why show:** see session guide
- **Input (truncated):**
  ```
  Use case name: DeleteTag 
  Description: Removes one or more search tags in the system. 
  Participating Actor: initialization Agency Operator 
  Entry Operator conditions: The agency has logged. 
  Flow of events User System: 
  1. Access the functionality to delete a tag. 
  2 Research in the existing system, the tags and displays them in a form. 
  3 Select one or more tags from the list and sends the request for deletion. 
  4 Delete the selected search tag. 
  Exit conditions: The system shall notify the successful elimination of selected tags. 
  Interruption of the connection to the server ETOUR. 
  Quality requirements:
  ```
- **Model output (raw):**
  ```
   
  - AdvancedSearch.java: implements part of the functionality to delete a tag.
  - AdvertisementAgencyManager.java: implements part of the functionality to delete a tag.
  - AdvertisementManager.java: implements part of the functionality to delete a tag.
  - AdvertisementRefreshmentPointManager.java: implements part of the functionality to delete a tag.
  - Authentication.java: not directly related to the use case.
  - BeanTag.java: contains the tag data model.
  - TagAgencyOperatorManager.java: manages the deletion of tags.
  - TagCommonManager.java: manages the deletion of tags.
  - Tag.java: contains the tag data model.
  - TagTableModel.java: displays the tags in a form.
  - DBTag.java: manages the deletion
  …
  ```
- **Gold:** `['BeanTag.java', 'IDBTag.java', 'ITagAgencyOperatorManager.java', 'TagAgencyOperatorManager.java', 'TagTableModel.java']`
- **Ask first (before gold):** Would your normal ~60s review catch/reject this?
