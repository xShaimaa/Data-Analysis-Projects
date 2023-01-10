/* Question 1: Who is the top customer? */
SELECT (Customer.FirstName || " " || Customer.LastName) as customer_name,
        SUM(Invoice.Total) AS total_spending
FROM Customer
JOIN Invoice 
ON Customer.CustomerId = Invoice.CustomerId
GROUP BY name
ORDER BY total_spending DESC
LIMIT 5

/* Question 2: Who is the best selling artist? */
SELECT Artist.ArtistId, Artist.Name AS artist_name, 
		SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) AS total_sales
FROM InvoiceLine
JOIN Track 
ON Track.TrackId = InvoiceLine.TrackId
JOIN Album 
ON Album.AlbumId = Track.AlbumId
JOIN Artist 
ON Artist.ArtistId = Album.ArtistId
GROUP BY Artist.ArtistId
ORDER BY 3 DESC
LIMIT 5
	
/* Question 3: What is the most frequent genre? */
SELECT Genre.Name AS genre_name, COUNT(Track.GenreId) AS Total
FROM Genre
JOIN Track 
ON Genre.GenreId = Track.GenreId
GROUP BY Genre.Name
ORDER BY COUNT(Track.GenreId) DESC
LIMIT 5

/* Question 4: Who is the top rock artist? */
SELECT DISTINCT Artist.ArtistId, Artist.Name AS artist_name, 
                COUNT(Genre.Name) AS rock_songs_count
FROM Artist
JOIN Album
ON Artist.ArtistId = Album.ArtistId
JOIN Track
ON Album.AlbumID = Track.AlbumId
JOIN Genre
ON Track.GenreId = Genre.GenreId
WHERE Genre.Name LIKE 'Rock'
GROUP BY Artist.ArtistId
ORDER BY COUNT(Genre.Name) DESC
LIMIT 10